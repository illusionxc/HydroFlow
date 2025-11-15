# --- START OF FILE rl_mapper_engine.py ---

import os
import argparse
import random
import time
from typing import Optional, Tuple, Dict, Any, List, Union
import numpy as np
import networkx as nx
import torch
import json
from rich import print
from rich.table import Table
from collections import defaultdict
from .build_activity_graph_online import build_activity_graph_online
from rl_algo.noc_utils import NoCUtils
from rl_algo.graph_partitioning import partition_graph
from rl_algo.NoCMappingEnv import NoCMappingEnv
from rl_algo.file_io import FilePathManager, save_mapping_results
from rl_algo.snn_feature_extractors import SNNFeatureExtractor



METRIC_KEY_MAP = {
    'composite_score': 'composite_score',
    'comm_cost': 'comm_cost',
    'average_weighted_hops': 'average_weighted_hops',
    'avg_link_load': 'avg_link_load',
    'num_links_used': 'num_links_used',
    'avg_packet_latency': 'avg_packet_latency',
    'total_energy_consumption': 'total_energy_consumption',
}

MAXIMIZATION_GOALS = {'throughput', 'saturation_throughput'}


try:
    from memory_profiler import memory_usage
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False
    def memory_usage(*args, **kwargs): return [0.0]
    
class RLMapperEngine:
    """
    
    """
    def __init__(self, args: argparse.Namespace):
        """
        
        """
        self.args = args
        self.device = torch.device(args.device)
        if args.device == 'cuda' and not torch.cuda.is_available():
            print(f"[yellow]Warning: CUDA requested but not available, falling back to CPU.[/yellow]")
            self.device = torch.device('cpu')

        self.file_manager = FilePathManager(args)
        self.snn_info: Optional[Dict[str, Any]] = None
        self.activity_graph: Optional[nx.Graph] = None
        self.evaluator_func: Optional[callable] = None        
        self.run_dir: Optional[str] = None
        self._partitioning_for_visualization: List[List[str]] = []
        self.last_training_results: Dict[str, Any] = {}



    def run(self):

        print(f"--- [bold]Starting RL NoC Mapping and Compilation Engine[/bold] ---")

        memory_snapshots = {}
        class MemoryMonitor:
            def __init__(self, stage_name):
                self.stage_name = stage_name
                self.start_mem, self.peak_mem = 0, 0
            def __enter__(self):
                if self.is_enabled():
                    mem_list = memory_usage(proc=-1, max_usage=False, timeout=1, interval=0.01)
                    self.start_mem = mem_list[-1] if mem_list else 0
                    print(f"  [dim]Entering stage '{self.stage_name}', Memory: {self.start_mem:.2f} MiB[/dim]")
                return self
            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.is_enabled():
                    peak_mem_val, _ = memory_usage((lambda: None,), max_usage=True, retval=True, timeout=1, interval=0.1)
                    end_mem_list = memory_usage(proc=-1, max_usage=False, timeout=1, interval=0.01)
                    end_mem = end_mem_list[-1] if end_mem_list else 0
                    self.peak_mem = max(self.start_mem, peak_mem_val, end_mem)
                    memory_snapshots[self.stage_name] = {'start': self.start_mem, 'end': end_mem, 'peak': self.peak_mem, 'delta': end_mem - self.start_mem}
                    print(f"  [dim]Completed stage '{self.stage_name}', Peak Memory: {self.peak_mem:.2f} MiB, End Memory: {end_mem:.2f} MiB[/dim]")
            def is_enabled(self):
                return self.args.enable_memory_profiling and MEMORY_PROFILER_AVAILABLE
        
        MemoryMonitor.args = self.args
        if self.args.enable_memory_profiling and not MEMORY_PROFILER_AVAILABLE:
            print("[bold red]Warning: Memory profiling requested, but 'memory-profiler' library is not installed.[/bold red]")
        
        run_dir_path = None
        training_results = {}
        components = None

        with MemoryMonitor("1. Graph Preparation"):
            self._prepare_graphs_and_info()
            assert self.activity_graph is not None, "Activity graph failed to load or build"

        with MemoryMonitor("2. Pre-computation & Feature Extraction"):
            print("  Pre-calculating global constants for environment instantiation...")
            node_activities = [float(data.get('total_spikes', 0.0)) for _, data in self.activity_graph.nodes(data=True)]
            self.max_neuron_activity = max(node_activities) if node_activities and max(node_activities) > 0 else 1.0
            from rl_algo.noc_utils import get_noc_core_coordinates
            self.core_coords_list = get_noc_core_coordinates((self.args.noc_rows, self.args.noc_cols))
            num_cores = self.args.noc_rows * self.args.noc_cols
            self.core_assignments_template = {i: [] for i in range(num_cores)}
            self.core_load_template = [0] * num_cores
            print(f"  [green]Global constants pre-calculation complete.[/green]")
            if self.args.gcpn_use_freq_features:
                detailed_spike_data = {'spike_times_per_sample': self.activity_graph.graph.get('spike_times_per_sample'), 'num_samples': self.activity_graph.graph.get('num_samples'), 'total_time_steps': self.activity_graph.graph.get('total_time_steps')}
                if all(detailed_spike_data.values()):
                    all_extracted_features = SNNFeatureExtractor.extract_all_features(graph=self.activity_graph, detailed_spike_data=detailed_spike_data, args=self.args)
                    if 'node_features' in all_extracted_features:
                        nx.set_node_attributes(self.activity_graph, all_extracted_features['node_features'], 'freq_feature')
                        print("[green]  ✅ Node spatio-temporal features calculated and attached to the activity graph.[/green]")
                    if 'edge_features' in all_extracted_features:
                        nx.set_edge_attributes(self.activity_graph, all_extracted_features['edge_features'])
                        print("[green]  ✅ Edge functional connectivity features calculated and attached to the activity graph.[/green]")
                else:
                    print("[yellow]  ⚠️ Warning: Detailed spike data is incomplete, skipping advanced spatio-temporal feature extraction.[/yellow]")


        is_hydroflow_direct_mode = self.args.mapping_mode == 'hierarchical' and \
                                  'hydroflow' in self.args.partitioning_algorithm.lower()


        if is_hydroflow_direct_mode:
                    analysis_history, vertex_order, int_to_node = None, np.array([]), {}

                    if not self.run_dir:
                        base_dir_for_algo = self.args.out_dir
                        algo_name = self.args.partitioning_algorithm.lower()
                        algo_dir = os.path.join(base_dir_for_algo, algo_name)
                        os.makedirs(algo_dir, exist_ok=True)
                        existing_runs = [int(d) for d in os.listdir(algo_dir) if d.isdigit() and os.path.isdir(os.path.join(algo_dir, d))]
                        run_number = max(existing_runs, default=0) + 1
                        self.run_dir = os.path.join(algo_dir, str(run_number))
                        os.makedirs(self.run_dir, exist_ok=True)
                        print(f"All outputs for HydroFlow-Mapping mode will be saved in: [green]{self.run_dir}[/green]")

                    with MemoryMonitor("3. HydroFlow Direct Compilation"):
                        print("\n[bold cyan]=== Running in HydroFlow-Mapping Integrated Compilation Mode ===[/bold cyan]")
                        
                        partitions_list, analysis_history, vertex_order, int_to_node = partition_graph(
                            full_graph=self.activity_graph,
                            core_capacity=self.args.core_capacity,
                            merge_threshold=self.args.component_merge_threshold,
                            args=self.args
                        )
                        
                        self._partitioning_for_visualization = partitions_list

                        
                        if self.args.visualize_hydroflow:
                            final_clusters_dict, self.macro_graph = self.partition_analyzer.analyze_characterize_and_visualize(
                                clusters_list=partitions_list,
                                full_snn_graph=self.activity_graph
                            )
                            print("  [green]XRL analysis complete, cluster descriptors and macro-graph generated.[/green]")
                    
                    with MemoryMonitor("4. Final Evaluation"):
                        final_neuron_mapping = {
                            neuron_id: core_id 
                            for core_id, neuron_list in enumerate(partitions_list) 
                            for neuron_id in neuron_list
                        }
                        
                        print("\n--- Step 3: Evaluating the final solution generated by HydroFlow-Mapping ---")
                        dummy_env = self._create_dummy_env()
                        final_metrics = self._evaluate_mirco_mapping(
                            mapping=final_neuron_mapping, 
                            graph=self.activity_graph, 
                            env=dummy_env
                        )
                        
                        primary_decision_key = METRIC_KEY_MAP.get(self.args.optimization_goal, 'composite_score')
                        is_maximization = self.args.optimization_goal in MAXIMIZATION_GOALS
                        best_obj_val = final_metrics.get(primary_decision_key, float('inf'))
                        if is_maximization and best_obj_val != float('inf'):
                            best_obj_val = -best_obj_val

                        training_results = {
                            "best_mapping": final_neuron_mapping,
                            "best_objective_value": best_obj_val,
                            "best_detailed_costs": final_metrics,
                            "history": {}
                        }
                        
                        self.env = dummy_env
                        self.agent = None

                        self.last_training_results = training_results



        with MemoryMonitor("5. Final Saving & Plotting"):
            if not self.run_dir:
                base_dir_for_algo = self.args.out_dir
                algo_name = self.args.partitioning_algorithm.lower()
                algo_dir = os.path.join(base_dir_for_algo, algo_name)
                os.makedirs(algo_dir, exist_ok=True)
                existing_runs = [int(d) for d in os.listdir(algo_dir) if d.isdigit() and os.path.isdir(os.path.join(algo_dir, d))]
                run_number = max(existing_runs, default=0) + 1
                self.run_dir = os.path.join(algo_dir, str(run_number))
                os.makedirs(self.run_dir, exist_ok=True)
                print(f"All outputs for HydroFlow-Mapping mode will be saved in: [green]{self.run_dir}[/green]")
            

            run_dir_path = self._save_and_plot_results(training_results, self.activity_graph, self.env)

        self._log_final_summary(run_dir_path, memory_snapshots, training_results)
        
        print(f"\n--- [bold]{self.args.rl_agent_type.upper()} NoC Mapping Engine run finished[/bold] ---")
        if run_dir_path:
            print(f"All output files have been saved to the directory: [green]{run_dir_path}[/green]")
    
            
    def _log_final_summary(self, run_dir_path: Optional[str], memory_snapshots: Dict[str, Dict[str, float]], training_results: Dict):
        """
        
        """
        print(f"\n--- [bold]{self.args.rl_agent_type.upper()} NoC Mapping Engine run finished[/bold] ---")
        
        if self.args.enable_memory_profiling and memory_snapshots:
            mem_table = Table(title="[bold]End-to-End Memory Usage (RSS)[/bold]", show_header=True, header_style="bold magenta")
            mem_table.add_column("Stage", style="dim", width=40)
            mem_table.add_column("Start", style="cyan", justify="right")
            mem_table.add_column("End", style="cyan", justify="right")
            mem_table.add_column("Delta", justify="right")
            mem_table.add_column("Peak in Stage", style="bold yellow", justify="right")
            
            overall_peak = 0
            for stage, data in sorted(memory_snapshots.items()):
                delta = data.get('delta', 0)
                delta_str = f"[green]+{delta:.2f}[/green]" if delta >= 0 else f"[red]{delta:.2f}[/red]"
                mem_table.add_row(
                    stage,
                    f"{data.get('start', 0):.2f} MiB",
                    f"{data.get('end', 0):.2f} MiB",
                    delta_str,
                    f"{data.get('peak', 0):.2f} MiB"
                )
                if data.get('peak', 0) > overall_peak:
                    overall_peak = data['peak']
            
            print(mem_table)
            print(f"[bold]Overall Peak Memory Detected: {overall_peak:.2f} MiB[/bold]")

        print("\n[bold]Final Best Mapping Results:[/bold]")
        best_costs = training_results.get('best_detailed_costs', {})
        if best_costs:
            for key, value in sorted(best_costs.items()):
                if isinstance(value, (int, float)):
                    print(f"  - {key.replace('_', ' ').title():<30}: {value:,.4f}")
        else:
            print("  [yellow]Could not find a valid final mapping solution.[/yellow]")
            
            

    def _create_dummy_env(self) -> NoCMappingEnv:
        """
        
        """
        if not hasattr(self, '_dummy_env_instance'):
            from rl_algo.noc_utils import get_noc_core_coordinates
            reward_config = self._configure_reward_config()
            ordered_neurons = self._get_ordered_neurons(self.activity_graph)
            
            if not hasattr(self, 'core_coords_list'):
                 self.core_coords_list = get_noc_core_coordinates((self.args.noc_rows, self.args.noc_cols))

            self._dummy_env_instance = NoCMappingEnv(
                activity_graph=self.activity_graph,
                neurons_to_place_ordered=ordered_neurons,
                noc_dims=(self.args.noc_rows, self.args.noc_cols),
                device=self.device,
                core_capacity=self.args.rl_core_capacity,
                reward_config=reward_config,
                max_neuron_activity=self.max_neuron_activity,
                core_coords_list=self.core_coords_list
            )
        return self._dummy_env_instance
    
            
    def _prepare_graphs_and_info(self):
        """
        
        """
        print("\n--- Step 1: Preparing SNN Activity Graph and Detailed Data ---")
        
        self.file_manager.load_summary()
        self.snn_info = self.file_manager.snn_info
        
        graph_source = getattr(self.args, 'graph_source', 'load_gexf')
        
        start_time = time.time()

        if graph_source == 'load_gexf':
            print("  Mode: Loading activity graph from pre-computed GEXF file...")
            self.activity_graph = self._load_or_build_activity_graph()
            
            print("  Loading detailed spike timing data for advanced feature engineering...")
            spike_times, num_samples, time_steps = self._load_detailed_spike_data_from_files()
            if spike_times:
                self.activity_graph.graph['spike_times_per_sample'] = spike_times
                self.activity_graph.graph['num_samples'] = num_samples
                self.activity_graph.graph['total_time_steps'] = time_steps
                print("  [green]Detailed spike data loaded and successfully attached to the graph.[/green]")

        elif graph_source == 'build_online':
            print(f"  [bold cyan]Mode: Building activity graph online from raw data (Strategy: {self.args.online_build_mode})...[/bold cyan]")
            
            self.activity_graph = build_activity_graph_online(
                args=self.args,
                file_manager=self.file_manager,
                snn_info=self.snn_info
            )
        
        else:
            print(f"  Mode: {graph_source} (will dynamically build and save GEXF)...")
            self.activity_graph = self._load_or_build_activity_graph()
            print("  Loading detailed spike timing data for advanced feature engineering...")
            spike_times, num_samples, time_steps = self._load_detailed_spike_data_from_files()
            if spike_times:
                self.activity_graph.graph['spike_times_per_sample'] = spike_times
                self.activity_graph.graph['num_samples'] = num_samples
                self.activity_graph.graph['total_time_steps'] = time_steps
                print("  [green]Detailed spike data loaded and successfully attached to the graph.[/green]")

        assert self.activity_graph is not None, "Activity graph failed to load or build"
        
        end_time = time.time()
        
        print(f" Online graph construction time: {end_time - start_time:.4f} seconds \n")


    def _evaluate_mirco_mapping(self, mapping: Dict[str, int], graph: nx.Graph, env: NoCMappingEnv) -> Dict[str, float]:
        """
        
        """
        if not mapping:
            return defaultdict(lambda: float('inf'))
            
        mapping_coords = {neuron_id: env.core_coords_list[core_idx] for neuron_id, core_idx in mapping.items()}
        
        sim_params = self._get_simulation_params()
        
        all_metrics = NoCUtils.calculate_all_noc_metrics(graph, mapping_coords, env.noc_dims, sim_params)

        composite_score = 0.0
        
        for key_arg, key_metric in METRIC_KEY_MAP.items():
            weight = getattr(self.args, f'metric_w_{key_arg}', 0.0)
            
            if weight != 0:
                metric_value = all_metrics.get(key_metric, 0.0)
                
                composite_score += weight * metric_value

        all_metrics['composite_score'] = composite_score
        
        return all_metrics
    
    
    def _save_and_plot_results(self, training_results: Dict[str, Any], graph: nx.Graph, env: Union[NoCMappingEnv]) -> Optional[str]:
        """
        
        """
        print("\n--- Step 4: Saving Results and Plotting Training Graphs ---")
        self.last_training_results = training_results
        
        run_dir = self.run_dir
        if not run_dir:
            print("[red]Error: run_dir was not initialized, cannot save results.[/red]")
            return None
        
        map_output_path = os.path.join(run_dir, "best_mapping_info.json")

        best_mapping_idx = training_results['best_mapping']
        
        if best_mapping_idx:
            best_mapping_coords = {nid: env.core_coords_list[cidx] for nid, cidx in best_mapping_idx.items()}
            save_mapping_results(
                output_path=map_output_path, 
                final_mapping_coords=best_mapping_coords, 
                activity_graph=graph,
                noc_dims=env.noc_dims, 
                snn_info=self.snn_info, 
                args=self.args,
                best_cost_metric=training_results['best_objective_value'], 
                detailed_costs=training_results['best_detailed_costs']
            )

        else:
            save_mapping_results(map_output_path, {}, graph, env.noc_dims, self.snn_info, self.args, float('inf'), {})

        return run_dir


    
    def _configure_reward_config(self) -> Dict[str, Any]:
        """
        
        """
        configs = {}
        args_dict = vars(self.args)

        reward_keys = [
            'rw_immediate_comm_weight', 'invalid_action_penalty', 
            'disable_immediate_reward', 
            'immediate_reward_strategy', 'k_neighbors_reward',
            'macro_reward_weight'  
        ]
        for key in reward_keys:
            if key in args_dict:
                configs[key] = args_dict[key]
        
        print("\n[bold]========== Reward and Evaluation Configuration Overview ==========[/bold]")
        
        print("  [bold]1. Immediate Reward/Penalty Configuration (for guiding the Agent):[/bold]")
        for k, v in configs.items():
            print(f"    - {k}: {v}")
        
        print("\n  [bold]2. Final Evaluation Weights (for calculating Composite Score):[/bold]")
        has_weights = False
        for key, value in args_dict.items():
            if key.startswith('metric_w_') and value != 0:
                print(f"    - {key}: {value}")
                has_weights = True
        if not has_weights:
            print("    [dim]No non-zero weights specified.[/dim]")

        print("\n  [bold]3. NoC Hardware Simulation Parameters:[/bold]")
        sim_param_keys = [
            'link_bandwidth', 'router_pipeline_delay', 'bits_per_flit', 'avg_packet_length',
            'energy_per_bit_link', 'energy_per_bit_router', 'burst_window_size'
        ]
        sim_params = self._get_simulation_params()
        for key in sim_param_keys:
            print(f"    - {key}: {sim_params[key]}")
        
        print("[bold]==========================================[/bold]")
        
        return configs
    

    def _get_simulation_params(self) -> Dict[str, Any]:
        """
        
        """
        return {
            'burst_window_size': self.args.burst_window_size,
            'link_bandwidth': self.args.link_bandwidth,
            'router_pipeline_delay': self.args.router_pipeline_delay,
            'avg_packet_length': self.args.avg_packet_length,
            'energy_per_bit_link': self.args.energy_per_bit_link,
            'energy_per_bit_router': self.args.energy_per_bit_router,
            'bits_per_flit': self.args.bits_per_flit
        }


                                
    def _get_ordered_neurons(self, graph: nx.Graph) -> List[str]:
        """
        
        """
        print("  Randomly sorting neurons (for placement order)...")
        
        neuron_list = list(graph.nodes())
        
        seed = getattr(self.args, 'seed', None)
        if seed is not None:
            print(f"  Using random seed {seed} for sorting.")
            seeded_random = random.Random(seed)
            seeded_random.shuffle(neuron_list)
        else:
            print("  No random seed provided, performing true random sorting.")
            random.shuffle(neuron_list)
            
        print(f"  Neuron random sorting complete, {len(neuron_list)} neurons to be mapped.")
        
        return neuron_list 
                                  
                
    def _load_detailed_spike_data_from_files(self) -> Tuple[Dict[int, Dict[int, List[int]]], Optional[int], Optional[int]]:
        """
        
        """
        summary_filepath = getattr(self.args, 'summary_filepath', None)
        spike_events_filepath = getattr(self.args, 'spike_events_filepath', None)

        if not summary_filepath:
            print("[bold red]Error: Analysis summary file path not provided via --summary_filepath argument.[/bold red]")
            return {}, None, None
        if not spike_events_filepath:
            print("[bold red]Error: Spike events file path not provided via --spike_events_filepath argument.[/bold red]")
            return {}, None, None
            
        if not os.path.exists(summary_filepath):
            print(f"[bold red]Error: Analysis summary file not found: {summary_filepath}[/bold red]")
            return {}, None, None
        if not os.path.exists(spike_events_filepath):
            print(f"[bold red]Error: Spike events file not found: {spike_events_filepath}[/bold red]")
            return {}, None, None

        num_samples = None
        total_time_steps = None
        
        try:
            with open(summary_filepath, 'r', encoding='utf-8') as f:
                summary_data = json.load(f)
            
            num_samples = summary_data.get('num_samples_analyzed')
            total_time_steps = summary_data.get('inference_time_steps')
            
            if not all([num_samples, total_time_steps]):
                print("[bold red]Error: Missing 'num_samples_analyzed' or 'inference_time_steps' key in the analysis summary file.[/bold red]")
                return {}, None, None

        except Exception as e:
            print(f"[bold red]Error reading or parsing analysis summary file '{summary_filepath}': {e}[/bold red]")
            return {}, None, None

        spike_times_per_sample: Dict[int, Dict[int, List[int]]] = defaultdict(lambda: defaultdict(list))

        try:
            with open(spike_events_filepath, "r", encoding="utf-8") as f_spikes:
                for line in f_spikes:
                    if line.startswith("#") or not line.strip(): continue
                    parts = line.strip().split(',')
                    if len(parts) == 3:
                        try:
                            time_step = int(parts[0]) - 1
                            sample_id = int(parts[1])
                            neuron_id = int(parts[2])
                            
                            if 0 <= time_step < total_time_steps and 0 <= sample_id < num_samples:
                                spike_times_per_sample[sample_id][neuron_id].append(time_step)
                        except (ValueError, IndexError):
                            continue 
        except Exception as e:
            print(f"[bold red]Error reading or processing spike events file '{spike_events_filepath}': {e}[/bold red]")
            return {}, num_samples, total_time_steps

        for sample_data in spike_times_per_sample.values():
            for spike_list in sample_data.values():
                spike_list.sort()

        return dict(spike_times_per_sample), num_samples, total_time_steps