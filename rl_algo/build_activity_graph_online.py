import argparse
from collections import defaultdict
import json
import os
import traceback
from typing import Dict, List, Any, Tuple, Iterator, Callable, Iterable
import gc

import networkx as nx
import numpy as np
from rich import print
from tqdm import tqdm
import numba

from joblib import Parallel, delayed, parallel_config

from rl_algo.graph_builders import (
    _get_layer_by_name,
    _get_neuron_shape_after_post_op,
)


@numba.njit(fastmath=True, cache=True)
def _calculate_joint_activity_kernel(
    node_chunk_indices: np.ndarray,
    static_adj_indices: np.ndarray,
    static_adj_list: np.ndarray,
    static_adj_weights: np.ndarray,
    spike_times_flat: np.ndarray,
    spike_indices_csr: np.ndarray,
    num_samples: int,
    use_static_weights: bool,
    response_delay_min: int,
    response_delay_max: int,
) -> np.ndarray:
    results = []
    for u in node_chunk_indices:
        start_edge_idx, end_edge_idx = static_adj_indices[u], static_adj_indices[u + 1]
        for edge_idx in range(start_edge_idx, end_edge_idx):
            v = static_adj_list[edge_idx]
            joint_strength_count = 0
            for sample_id in range(num_samples):
                u_start, u_end = spike_indices_csr[sample_id, u, 0], spike_indices_csr[sample_id, u, 1]
                v_start, v_end = spike_indices_csr[sample_id, v, 0], spike_indices_csr[sample_id, v, 1]
                if u_start < u_end and v_start < v_end:
                    source_spikes = spike_times_flat[u_start:u_end]
                    target_spikes = spike_times_flat[v_start:v_end]
                    idx_v = 0
                    for t_u in source_spikes:
                        while idx_v < len(target_spikes) and target_spikes[idx_v] < t_u + response_delay_min:
                            idx_v += 1
                        match_idx = idx_v
                        while match_idx < len(target_spikes):
                            t_v = target_spikes[match_idx]
                            delay = t_v - t_u
                            if response_delay_min <= delay <= response_delay_max:
                                joint_strength_count += 1
                            elif delay > response_delay_max:
                                break
                            match_idx += 1
            if joint_strength_count > 0:
                final_strength = float(joint_strength_count)
                if use_static_weights:
                    final_strength *= abs(static_adj_weights[edge_idx])
                if final_strength > 1e-9:
                    results.append((u, v, final_strength))
    if not results:
        return np.empty((0, 3), dtype=np.float32)
    return np.array(results, dtype=np.float32)

def _iterate_physical_edges_from_layers(
    layer_chunk: List[Dict[str, Any]],
    topology: Dict[str, Any],
    weights_dir: str
) -> Iterator[Tuple[int, int, float]]:
    for target_layer_info in layer_chunk:
        driving_layers = target_layer_info.get('driving_layers', []) or ([target_layer_info.get('driving_layer')] if target_layer_info.get('driving_layer') else [])
        for driving_layer in driving_layers:
            source_layer_name = driving_layer.get('source')
            if not source_layer_name or source_layer_name == "input": continue
            source_layer_info = _get_layer_by_name(topology, source_layer_name)
            if not source_layer_info: continue
            weights_path = os.path.join(weights_dir, driving_layer['weights_file'])
            if not os.path.exists(weights_path): continue
            weights = np.load(weights_path)
            source_start_idx = source_layer_info.get('global_start_index')
            target_start_idx = target_layer_info.get('global_start_index')
            if not all(isinstance(idx, int) and idx >= 0 for idx in [source_start_idx, target_start_idx]): continue
            if driving_layer.get('type') == 'Linear':
                num_target, num_source = weights.shape
                for j in range(num_source):
                    u = source_start_idx + j
                    for i in range(num_target):
                        weight = weights[i, j]
                        if abs(weight) > 1e-9:
                            v = target_start_idx + i
                            yield u, v, weight
            elif driving_layer.get('type') == 'Conv2d':
                params = driving_layer.get('params', {})
                source_shape = _get_neuron_shape_after_post_op(source_layer_info)
                target_shape = tuple(target_layer_info.get('neuron_shape'))
                if not all([source_shape, target_shape, len(source_shape) >= 3, len(target_shape) >= 3]): continue
                C_in, H_in, W_in = source_shape
                C_out, H_out, W_out = target_shape
                k_size = params.get('kernel_size', (3, 3)); k_h, k_w = (k_size, k_size) if isinstance(k_size, int) else k_size
                stride = params.get('stride', (1, 1)); s_h, s_w = (stride, stride) if isinstance(stride, int) else stride
                padding = params.get('padding', (0, 0)); p_h, p_w = (padding, padding) if isinstance(padding, int) else padding
                for c_out in range(C_out):
                    for h_out in range(H_out):
                        for w_out in range(W_out):
                            v = target_start_idx + c_out * H_out * W_out + h_out * W_out + w_out
                            for c_in in range(C_in):
                                for kh in range(k_h):
                                    for kw in range(k_w):
                                        h_in = h_out * s_h - p_h + kh
                                        w_in = w_out * s_w - p_w + kw
                                        if 0 <= h_in < H_in and 0 <= w_in < W_in:
                                            weight = weights[c_out, c_in, kh, kw]
                                            if abs(weight) > 1e-9:
                                                u = source_start_idx + c_in * H_in * W_in + h_in * W_in + w_in
                                                yield u, v, weight

def _calculate_degrees_chunk(
    layer_chunk: List[Dict[str, Any]],
    topology: Dict[str, Any],
    weights_dir: str,
    total_lif_neurons: int
) -> np.ndarray:
    local_degrees = np.zeros(total_lif_neurons, dtype=np.int32)
    edge_iterator = _iterate_physical_edges_from_layers(layer_chunk, topology, weights_dir)
    for u, _, _ in edge_iterator:
        local_degrees[u] += 1
    return local_degrees

def _fill_csr_chunk(
    layer_chunk: List[Dict[str, Any]],
    topology: Dict[str, Any],
    weights_dir: str
) -> Dict[int, List[Tuple[int, float]]]:
    local_edges_by_source = defaultdict(list)
    edge_iterator = _iterate_physical_edges_from_layers(layer_chunk, topology, weights_dir)
    for u, v, w in edge_iterator:
        local_edges_by_source[u].append((v, w))
    return dict(local_edges_by_source)


def _build_static_csr_from_topology(
    topology: Dict[str, Any],
    weights_dir: str,
    total_lif_neurons: int,
    n_jobs: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    print("CSR building...")
    
    layers = topology.get('layers', [])
    num_chunks = n_jobs * 2
    chunk_size = (len(layers) + num_chunks - 1) // num_chunks
    layer_chunks = [layers[i:i + chunk_size] for i in range(0, len(layers), chunk_size) if i < len(layers)]

    print("        Pass 1/2: parallel degree calculation...")
    tasks_pass1 = [delayed(_calculate_degrees_chunk)(
        chunk, topology, weights_dir, total_lif_neurons
    ) for chunk in layer_chunks]
    
    with parallel_config(backend="loky", temp_folder="/tmp/joblib_temp"):
        p = Parallel(n_jobs=n_jobs, batch_size="auto")
        results_pass1 = p(
            tqdm(tasks_pass1, desc="          Parallel scanning layer chunks (Pass 1)", ncols=100)
        )

    degrees = np.sum(results_pass1, axis=0, dtype=np.int32)
    total_edges = int(degrees.sum())
    del results_pass1; gc.collect()
    
    print("        Pass 2/2: parallel filling of CSR arrays...")
    adj_indices = np.zeros(total_lif_neurons + 1, dtype=np.int32)
    adj_indices[1:] = np.cumsum(degrees)
    adj_list = np.zeros(total_edges, dtype=np.int32)
    adj_weights = np.zeros(total_edges, dtype=np.float32)
    
    print("          - Step 2.1: parallel collect edges...")
    tasks_pass2_collect = [delayed(_fill_csr_chunk)(
        chunk, topology, weights_dir
    ) for chunk in layer_chunks]
    
    with parallel_config(backend="loky", temp_folder="/tmp/joblib_temp"):
        p = Parallel(n_jobs=n_jobs, batch_size="auto")
        results_pass2 = p(
            tqdm(tasks_pass2_collect, desc="            Parallel edge collection", ncols=100)
        )

    print("          - Step 2.2: serial fill data...")
    current_pos = np.copy(adj_indices[:-1])
    for local_edges_by_source in tqdm(results_pass2, desc="            Filling data", ncols=100):
        for u, edges in local_edges_by_source.items():
            num_edges = len(edges)
            start_idx = current_pos[u]
            end_idx = start_idx + num_edges
            v_list, w_list = zip(*edges)
            adj_list[start_idx:end_idx] = v_list
            adj_weights[start_idx:end_idx] = w_list
            current_pos[u] = end_idx
            
    del results_pass2; gc.collect()

    print("      - [CSR build] Physical connections CSR construction completed.")
    return adj_indices, adj_list, adj_weights




def _prepare_spike_data_for_numba(
    spike_times_per_sample: Dict[int, Dict[int, List[int]]],
    num_samples: int,
    total_lif_neurons: int
) -> Tuple[np.ndarray, np.ndarray]:

    for sample_data in spike_times_per_sample.values():
        for spike_list in sample_data.values():
            spike_list.sort()
    total_spikes_count = sum(len(times) for sample_data in spike_times_per_sample.values() for times in sample_data.values())
    spike_times_flat = np.zeros(total_spikes_count, dtype=np.int32)
    spike_indices_csr = np.zeros((num_samples, total_lif_neurons, 2), dtype=np.int32)
    current_pos = 0
    for sample_id in tqdm(range(num_samples), desc="       convert", ncols=100):
        sample_data = spike_times_per_sample.get(sample_id, {})
        for neuron_id in range(total_lif_neurons):
            spike_times = sample_data.get(neuron_id, [])
            num_spikes = len(spike_times)
            spike_indices_csr[sample_id, neuron_id, 0] = current_pos
            if num_spikes > 0:
                spike_times_flat[current_pos : current_pos + num_spikes] = spike_times
                current_pos += num_spikes
            spike_indices_csr[sample_id, neuron_id, 1] = current_pos
    return spike_times_flat, spike_indices_csr

def _build_dict_chunk(items_chunk: List[Any], key_func: Callable, value_func: Callable) -> Dict:
    return {key_func(item): value_func(item) for item in items_chunk}

def _parallel_dict_builder(
    items: Iterable,
    key_func: Callable,
    value_func: Callable,
    n_jobs: int,
    desc: str
) -> Dict:
    items_list = list(items)
    if not items_list:
        return {}
    num_chunks = n_jobs * 4
    chunk_size = (len(items_list) + num_chunks - 1) // num_chunks
    item_chunks = [items_list[i:i + chunk_size] for i in range(0, len(items_list), chunk_size)]
    tasks = [delayed(_build_dict_chunk)(
        chunk, key_func, value_func
    ) for chunk in item_chunks if chunk]
    with parallel_config(backend="loky", temp_folder="/tmp/joblib_temp"):
        sub_dicts = Parallel(n_jobs=n_jobs)(
            tqdm(tasks, desc=f"          {desc}", ncols=100)
        )
    final_dict = {}
    for sub_dict in tqdm(sub_dicts, desc="          merging", ncols=100):
        final_dict.update(sub_dict)
    del sub_dicts; gc.collect()
    return final_dict

def _build_graph_chunk(
    items_chunk: List[Any],
    edge_generator_func: Callable,
    **kwargs
) -> nx.DiGraph:
    local_graph = nx.DiGraph()
    edge_iterator = edge_generator_func(items_chunk, **kwargs)
    local_graph.add_edges_from(edge_iterator)
    return local_graph

def _parallel_graph_builder(
    items_to_process: List[Any],
    edge_generator_func: Callable,
    n_jobs: int,
    desc: str,
    **kwargs
) -> nx.DiGraph:

    num_chunks = n_jobs * 4
    chunk_size = (len(items_to_process) + num_chunks - 1) // num_chunks
    item_chunks = [items_to_process[i:i + chunk_size] for i in range(0, len(items_to_process), chunk_size)]
    tasks = [delayed(_build_graph_chunk)(
        chunk, edge_generator_func, **kwargs
    ) for chunk in item_chunks if chunk]
    with parallel_config(backend="loky", temp_folder="/tmp/joblib_temp"):
        sub_graphs = Parallel(n_jobs=n_jobs)(
            tqdm(tasks, desc=f"          {desc}", ncols=100)
        )

    final_graph = nx.DiGraph()
    for sub_graph in tqdm(sub_graphs, desc="          merging", ncols=100):
        final_graph.update(edges=sub_graph.edges(data=True), nodes=sub_graph.nodes(data=True))
    del sub_graphs; gc.collect()
    return final_graph

def _build_based_on_peak_activity(
    peak_strategy_name: str, 
    args: argparse.Namespace, 
    file_manager: Any, 
    snn_info: Dict[str, Any]
) -> nx.DiGraph:

    try:
        num_workers = os.cpu_count() or 1
        
        topology_path = file_manager.get_path("topology")
        with open(topology_path, 'r', encoding='utf-8') as f:
            topology = json.load(f)
        total_lif_neurons = snn_info.get('total_lif_neurons')
        num_samples = snn_info.get('num_samples_analyzed')
        total_time_steps = snn_info.get('inference_time_steps')
        if not all([total_lif_neurons, num_samples, total_time_steps]):
            raise ValueError("lack of meta data in summary file.")
        spike_events_path = file_manager.get_path("spike_events")
        spike_times_per_sample = defaultdict(lambda: defaultdict(list))
        activity_map = defaultdict(float)
        print("      - reading spike events...")
        with open(spike_events_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("#") or not line.strip(): continue
                try:
                    time_step, sample_id, neuron_id = map(int, line.strip().split(','))
                    if 0 <= sample_id < num_samples and 0 <= (time_step - 1) < total_time_steps:
                        spike_times_per_sample[sample_id][neuron_id].append(time_step - 1)
                        activity_map[neuron_id] += 1
                except (IndexError, ValueError):
                    continue
        spike_times_per_sample = dict(spike_times_per_sample)
        active_neurons_set = set(activity_map.keys())
        
        weights_dir = os.path.dirname(topology_path)
        static_adj_indices, static_adj_list, static_adj_weights = _build_static_csr_from_topology(
            topology, weights_dir, total_lif_neurons, n_jobs=num_workers
        )
        spike_times_flat, spike_indices_csr = _prepare_spike_data_for_numba(
            spike_times_per_sample, num_samples, total_lif_neurons
        )
        print("    Step 3/4: parallel computing joint activity...")
        node_indices_for_js = np.arange(total_lif_neurons, dtype=np.int32)
        node_chunks_for_js = np.array_split(node_indices_for_js, num_workers * 4)
        tasks_js = [delayed(_calculate_joint_activity_kernel)(
            chunk, static_adj_indices, static_adj_list, static_adj_weights,
            spike_times_flat, spike_indices_csr, num_samples,
            args.use_static_weights_in_joint_strength,
            args.response_delay_min, args.response_delay_max
        ) for chunk in node_chunks_for_js]
        with parallel_config(backend="loky", temp_folder="/tmp/joblib_temp"):
            results_list = Parallel(n_jobs=num_workers, pre_dispatch='2*n_jobs')(
                tqdm(tasks_js, desc="      compute joint actitvy", ncols=100, unit="chunk")
            )
        
        valid_results = [res for res in results_list if res is not None and res.shape[0] > 0]
        if not valid_results:
            joint_activity_results = np.empty((0, 3), dtype=np.float32)
        else:
            joint_activity_results = np.concatenate(valid_results)
        
        del results_list, spike_times_flat, spike_indices_csr; gc.collect()


        print("    step 4/4: constructing final graph and attaching all properties...")
        joint_strength_map = _parallel_dict_builder(
            items=joint_activity_results,
            key_func=lambda item: (int(item[0]), int(item[1])),
            value_func=lambda item: item[2],
            n_jobs=num_workers,
            desc="joint_strength_map"
        )
        del joint_activity_results; gc.collect()

        print(f"      - stratgy 'peak_{peak_strategy_name}'")
        
        if peak_strategy_name == 'static_full_topo':
            def edge_generator_static_full(nodes_to_process: List[int], **kwargs):
                adj_indices, adj_list, adj_weights = kwargs['csr_data']
                js_map = kwargs['js_map']
                act_map = kwargs['act_map']
                for u in nodes_to_process:
                    source_activity = act_map.get(u, 0.0)
                    for edge_idx in range(adj_indices[u], adj_indices[u+1]):
                        v = adj_list[edge_idx]
                        yield (u, v, {
                            'weight': float(adj_weights[edge_idx]),
                            'joint_activity_strength': js_map.get((u, v), 0.0),
                            'source_activity': source_activity
                        })
            
            all_nodes = list(range(total_lif_neurons))
            final_graph = _parallel_graph_builder(
                all_nodes, edge_generator_static_full, num_workers, "constructing static_full_topo",
                csr_data=(static_adj_indices, static_adj_list, static_adj_weights),
                js_map=joint_strength_map,
                act_map=activity_map
            )
            if final_graph.number_of_nodes() < total_lif_neurons:
                final_graph.add_nodes_from(range(total_lif_neurons))

        elif peak_strategy_name in ['dynamic_topo', 'static2dynamic_topo']:
            def edge_generator_dynamic_topo(items_chunk: List[Tuple], **kwargs):
                adj_indices, adj_list, adj_weights = kwargs['csr_data']
                act_map = kwargs['act_map']
                for u, v, js in items_chunk:
                    weight = 1.0 
                    start_idx, end_idx = adj_indices[u], adj_indices[u+1]
                    neighbors = adj_list[start_idx:end_idx]
                    match_indices = np.where(neighbors == v)[0]
                    if len(match_indices) > 0:
                        weight = adj_weights[start_idx + match_indices[0]]
                    yield (u, v, {
                        'weight': float(weight),
                        'joint_activity_strength': float(js),
                        'source_activity': act_map.get(u, 0.0)
                    })

            items_for_dynamic_build = [(u, v, js) for (u, v), js in joint_strength_map.items()]
            final_graph = _parallel_graph_builder(
                items_for_dynamic_build, edge_generator_dynamic_topo, num_workers, "Building functional edge chunks (with attributes)",
                csr_data=(static_adj_indices, static_adj_list, static_adj_weights),
                act_map=activity_map
            )
        else:
            raise ValueError(f"Unknown peak activity build strategy: '{peak_strategy_name}'")
        
        print("        - Attaching node attributes...")
        total_spikes_dict = {node: activity_map.get(node, 0.0) for node in final_graph.nodes()}
        nx.set_node_attributes(final_graph, total_spikes_dict, 'total_spikes')
        del total_spikes_dict
        
        is_active_dict = {node: (node in active_neurons_set) for node in final_graph.nodes()}
        nx.set_node_attributes(final_graph, is_active_dict, 'is_active')
        del is_active_dict; gc.collect()
        
        return _finalize_graph(final_graph, snn_info, spike_times_per_sample)
    
    except Exception as e:
        print(f"[bold red]Error: Graph construction based on peak activity failed: {e}[/bold red]")
        traceback.print_exc()
        return nx.DiGraph()

def _build_based_on_individual_activity(
    act_strategy_name: str, 
    args: argparse.Namespace, 
    file_manager: Any, 
    snn_info: Dict[str, Any]
) -> nx.DiGraph:
    try:
        num_workers = os.cpu_count() or 1
        print("    Step 1/3: Loading topology and efficiently computing individual activities...")
        topology_path = file_manager.get_path("topology")
        with open(topology_path, 'r', encoding='utf-8') as f:
            topology = json.load(f)
        total_lif_neurons = snn_info.get('total_lif_neurons')
        if not total_lif_neurons:
             raise ValueError("Missing 'total_lif_neurons' in SNN summary file.")

        spike_events_path = file_manager.get_path("spike_events")
        
        print("      - [NumPy acceleration] Reading and processing spike events to compute individual activities...")
        activity_counts = np.zeros(total_lif_neurons, dtype=np.int32)
        try:
            spike_data = np.loadtxt(spike_events_path, delimiter=',', usecols=(2,), dtype=np.int32, comments='#')
            if spike_data.size > 0:
                valid_spikes = spike_data[spike_data < total_lif_neurons]
                if valid_spikes.size > 0:
                    counts = np.bincount(valid_spikes)
                    if len(counts) > 0:
                        activity_counts[:len(counts)] += counts
        except Exception as e:
            print(f"      [yellow]Warning: NumPy fast loading failed ({e}), falling back to line-by-line reading.[/yellow]")
            with open(spike_events_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("#") or not line.strip(): continue
                    try:
                        neuron_id = int(line.strip().split(',')[2])
                        if 0 <= neuron_id < total_lif_neurons:
                            activity_counts[neuron_id] += 1
                    except (IndexError, ValueError):
                        continue

        active_neurons_set = set(np.where(activity_counts > 0)[0])
        activity_map = {i: count for i, count in enumerate(activity_counts) if count > 0}
        
        print(f"    Data loading complete, found {len(active_neurons_set)} active neurons.")
        
        print("    Step 2/3: Building graph based on selected strategy...")
        final_graph = nx.DiGraph()
        weights_dir = os.path.dirname(topology_path)
        
        if act_strategy_name in ['static_full_topo', 'static2dynamic_topo']:
            print(f"[bold cyan]      Strategy '{act_strategy_name}': Building with CSR acceleration...[/bold cyan]")
            adj_indices, adj_list, adj_weights = _build_static_csr_from_topology(
                topology, weights_dir, total_lif_neurons, n_jobs=num_workers
            )

            def edge_generator_with_act_attr(nodes_to_process: List[int], **kwargs):
                adj_indices, adj_list, adj_weights = kwargs['csr_data']
                act_map = kwargs['act_map']
                node_filter = kwargs.get('node_filter')

                for u in nodes_to_process:
                    source_activity = act_map.get(u, 0.0)
                    for i in range(adj_indices[u], adj_indices[u+1]):
                        v = int(adj_list[i])
                        if node_filter and v not in node_filter:
                            continue
                        yield u, v, {
                            'weight': float(adj_weights[i]),
                            'source_activity': source_activity
                        }

            if act_strategy_name == 'static_full_topo':
                all_nodes = list(range(total_lif_neurons))
                final_graph = _parallel_graph_builder(
                    all_nodes, edge_generator_with_act_attr, num_workers, "Building physical edge chunks (with attributes)",
                    csr_data=(adj_indices, adj_list, adj_weights),
                    act_map=activity_map
                )
                if final_graph.number_of_nodes() < total_lif_neurons:
                    final_graph.add_nodes_from(range(total_lif_neurons))
            
            elif act_strategy_name == 'static2dynamic_topo':
                active_nodes_list = list(active_neurons_set)
                final_graph = _parallel_graph_builder(
                    active_nodes_list, edge_generator_with_act_attr, num_workers, "Building active edge chunks (with attributes)",
                    csr_data=(adj_indices, adj_list, adj_weights),
                    act_map=activity_map,
                    node_filter=active_neurons_set
                )
                final_graph.add_nodes_from(active_nodes_list)

        elif act_strategy_name == 'dynamic_topo':
            print("[bold cyan]      Strategy 'act_dynamic_topo' (parallel optimized):[/bold cyan]")
            
            layers = topology.get('layers', [])
            num_chunks = num_workers * 2
            chunk_size = (len(layers) + num_chunks - 1) // num_chunks
            layer_chunks = [layers[i:i + chunk_size] for i in range(0, len(layers), chunk_size) if i < len(layers)]

            tasks = [delayed(_process_layer_chunk_for_dynamic_topo)(
                chunk, topology, weights_dir, active_neurons_set, activity_map
            ) for chunk in layer_chunks]

            with parallel_config(backend="loky", temp_folder="/tmp/joblib_temp"):
                p = Parallel(n_jobs=num_workers, batch_size="auto")
                list_of_edge_lists = p(tqdm(tasks, desc="          Parallel scanning layer chunks", ncols=100))
            
            import itertools
            all_edges = list(itertools.chain.from_iterable(list_of_edge_lists))
            
            print("        - Building final graph...")
            final_graph.add_nodes_from(list(active_neurons_set))
            if all_edges:
                final_graph.add_edges_from(tqdm(all_edges, desc="          Adding active edges", ncols=100))

        else:
             raise ValueError(f"Unknown individual activity build strategy: '{act_strategy_name}'")

        print("        - Attaching node attributes...")
        total_spikes_dict = {node: activity_map.get(node, 0.0) for node in final_graph.nodes()}
        nx.set_node_attributes(final_graph, total_spikes_dict, 'total_spikes')

        return _finalize_graph(final_graph, snn_info)

    except Exception as e:
        print(f"[bold red]Error: Graph construction based on individual activity failed: {e}[/bold red]")
        traceback.print_exc()
        return nx.DiGraph()

def _finalize_graph(
    graph: nx.DiGraph, 
    snn_info: Dict[str, Any], 
    spike_times_per_sample: Dict = None
) -> nx.DiGraph:
    print(f"    Finalizing: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges...")
    mapping = {node_id: str(node_id) for node_id in graph.nodes()}
    string_keyed_graph = nx.relabel_nodes(graph, mapping)
    string_keyed_graph.graph['spike_times_per_sample'] = spike_times_per_sample or {}
    string_keyed_graph.graph['num_samples'] = snn_info.get('num_samples_analyzed')
    string_keyed_graph.graph['total_time_steps'] = snn_info.get('inference_time_steps')
    print("    [green]Graph construction complete.[/green]")
    return string_keyed_graph

def build_activity_graph_online(
    args: argparse.Namespace, 
    file_manager: Any, 
    snn_info: Dict[str, Any]
) -> nx.DiGraph:
    build_strategy = args.online_build_mode
    print(f"\n[bold cyan]====== Online Graph Construction Started (Strategy: {build_strategy}) ======[/bold cyan]")
    if build_strategy.startswith('peak_'):
        peak_strategy_name = build_strategy.replace('peak_', '', 1)
        return _build_based_on_peak_activity(
            peak_strategy_name, args, file_manager, snn_info
        )
    elif build_strategy.startswith('act_'):
        act_strategy_name = build_strategy.replace('act_', '', 1)
        return _build_based_on_individual_activity(
            act_strategy_name, args, file_manager, snn_info
        )
    else:
        raise ValueError(f"Unknown online build strategy: '{build_strategy}'. Must start with 'peak_' or 'act_'.")
    
    
    
    
    
def _generate_fc_edges(
    source_layer_info: Dict, target_layer_info: Dict, weights: np.ndarray, 
    active_neurons_set: set, activity_map: Dict
) -> Iterator[Tuple[int, int, Dict]]:
    source_start = source_layer_info['global_start_index']
    target_start = target_layer_info['global_start_index']
    num_target, num_source = weights.shape

    for j in range(num_source):
        u = source_start + j
        if u in active_neurons_set:
            source_activity = activity_map.get(u, 0.0)
            for i in range(num_target):
                v = target_start + i
                yield u, v, {'weight': float(weights[i, j]), 'source_activity': source_activity}

def _generate_conv_edges(
    source_layer_info: Dict, target_layer_info: Dict, weights: np.ndarray, 
    params: Dict, source_shape: Tuple, target_shape: Tuple,
    active_neurons_set: set, activity_map: Dict
) -> Iterator[Tuple[int, int, Dict]]:
    source_start = source_layer_info['global_start_index']
    target_start = target_layer_info['global_start_index']
    C_in, H_in, W_in = source_shape
    C_out, H_out, W_out = target_shape
    k_size = params.get('kernel_size', (3, 3)); k_h, k_w = (k_size, k_size) if isinstance(k_size, int) else k_size
    stride = params.get('stride', (1, 1)); s_h, s_w = (stride, stride) if isinstance(stride, int) else stride
    padding = params.get('padding', (0, 0)); p_h, p_w = (padding, padding) if isinstance(padding, int) else padding
    
    for c_out in range(C_out):
        for h_out in range(H_out):
            for w_out in range(W_out):
                v = target_start + c_out * H_out * W_out + h_out * W_out + w_out
                if v in active_neurons_set:
                    for c_in in range(C_in):
                        for kh in range(k_h):
                            for kw in range(k_w):
                                h_in = h_out * s_h - p_h + kh
                                w_in = w_out * s_w - p_w + kw
                                if 0 <= h_in < H_in and 0 <= w_in < W_in:
                                    u = source_start + c_in * H_in * W_in + h_in * W_in + w_in
                                    if u in active_neurons_set and abs(weights[c_out, c_in, kh, kw]) > 1e-9:
                                        source_activity = activity_map.get(u, 0.0)
                                        yield u, v, {'weight': float(weights[c_out, c_in, kh, kw]), 'source_activity': source_activity}


def _process_layer_chunk_for_dynamic_topo(
    layer_chunk: List[Dict[str, Any]],
    topology: Dict[str, Any],
    weights_dir: str,
    active_neurons_set: set,
    activity_map: Dict[int, float]
) -> List[Tuple[int, int, Dict]]:
    edges_found = []
    for target_layer_info in layer_chunk:
        target_start = target_layer_info.get('global_start_index', -1)
        if target_start == -1 or not any(n in active_neurons_set for n in range(target_start, target_start + target_layer_info.get('neuron_count_flat', 0))):
            continue

        driving_layers = target_layer_info.get('driving_layers', []) or ([target_layer_info.get('driving_layer')] if target_layer_info.get('driving_layer') else [])
        for driving_layer in driving_layers:
            source_layer_name = driving_layer.get('source')
            if not source_layer_name or source_layer_name == "input": continue
            
            source_layer_info = _get_layer_by_name(topology, source_layer_name)
            if not source_layer_info: continue
            
            source_start = source_layer_info.get('global_start_index', -1)
            if source_start == -1 or not any(n in active_neurons_set for n in range(source_start, source_start + source_layer_info.get('neuron_count_flat', 0))):
                continue
                
            weights_path = os.path.join(weights_dir, driving_layer['weights_file'])
            if not os.path.exists(weights_path): continue
            weights = np.load(weights_path)
            driving_type = driving_layer.get('type')

            if driving_type == 'Linear':
                edges_found.extend(
                    _generate_fc_edges(source_layer_info, target_layer_info, weights, active_neurons_set, activity_map)
                )
            elif driving_type == 'Conv2d':
                source_shape_op = _get_neuron_shape_after_post_op(source_layer_info)
                target_shape = tuple(target_layer_info.get('neuron_shape'))
                if all([source_shape_op, target_shape, len(source_shape_op) >= 3, len(target_shape) >= 3]):
                    edges_found.extend(
                        _generate_conv_edges(source_layer_info, target_layer_info, weights, driving_layer.get('params', {}), 
                                             source_shape_op, target_shape, active_neurons_set, activity_map)
                    )
    return edges_found