import os
from joblib import Parallel, delayed
import networkx as nx
import numpy as np
from typing import Dict, Tuple, List, Optional
from collections import defaultdict
import torch.jit as jit
import torch
try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


@numba.njit(fastmath=True, cache=True)
def get_noc_core_coordinates(noc_dims: tuple[int, int]) -> list[tuple[int, int]]:
    rows, cols = noc_dims
    return [(r, c) for r in range(rows) for c in range(cols)]

@numba.njit(fastmath=True, cache=True)
def _get_xy_routing_path_numba(start_coord, end_coord):
    path = [(start_coord[0], start_coord[1])]
    curr_r, curr_c = start_coord
    end_r, end_c = end_coord
    while curr_c != end_c:
        curr_c += 1 if end_c > curr_c else -1
        path.append((curr_r, curr_c))
    while curr_r != end_r:
        curr_r += 1 if end_r > curr_r else -1
        path.append((curr_r, curr_c))
    return path

@jit.script
def manhattan_distance_torch(coords1: torch.Tensor, coords2: torch.Tensor) -> torch.Tensor:
    diff = coords1.unsqueeze(-2) - coords2.unsqueeze(-3)
    return torch.abs(diff).sum(dim=-1)

@numba.njit(fastmath=True, cache=True)
def manhattan_distance(coord1: tuple[int, int], coord2: tuple[int, int]) -> int:
    return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])


def _preprocess_graph_edges_for_comm(activity_graph, mapping_coords, coord_to_idx):
    inter_core_edges = []
    for u, v, data in activity_graph.edges(data=True):
        if u in mapping_coords and v in mapping_coords:
            start_coord, end_coord = mapping_coords[u], mapping_coords[v]
            if start_coord in coord_to_idx and end_coord in coord_to_idx and start_coord != end_coord:
                inter_core_edges.append((
                    coord_to_idx[start_coord],
                    coord_to_idx[end_coord],
                    float(data.get('source_activity', 0.0))
                ))
    if not inter_core_edges:
        return None
    
    edge_data = np.array(inter_core_edges, dtype=np.float32)
    src_indices = torch.from_numpy(edge_data[:, 0]).long()
    dst_indices = torch.from_numpy(edge_data[:, 1]).long()
    volumes = torch.from_numpy(edge_data[:, 2]).float()
    return src_indices, dst_indices, volumes


def _build_routing_sub_matrix(source_core_chunk, num_cores, core_coords, link_to_idx, num_links) -> torch.Tensor:
    sub_matrix = torch.zeros((len(source_core_chunk), num_cores, num_links), dtype=torch.float32)
    for i_chunk, i_global in enumerate(source_core_chunk):
        for j in range(num_cores):
            if i_global == j: continue
            path = NoCUtils._get_xy_routing_path(core_coords[i_global], core_coords[j])
            for hop in range(len(path) - 1):
                link = (path[hop], path[hop+1])
                if link in link_to_idx:
                    sub_matrix[i_chunk, j, link_to_idx[link]] = 1.0
    return sub_matrix

@numba.njit(fastmath=True, cache=True)
def _calculate_latency_worker_numba(
    edges_info: np.ndarray, path_matrix: np.ndarray, path_lengths: np.ndarray,
    link_loads_array: np.ndarray, router_delay: int, link_bandwidth: float,
    avg_packet_length: int, saturated_penalty: float, total_time_steps: int
) -> np.ndarray:
    latencies = np.zeros(len(edges_info), dtype=np.float32)
    for i in range(len(edges_info)):
        src_idx, dst_idx = int(edges_info[i, 0]), int(edges_info[i, 1])
        hops = path_lengths[src_idx, dst_idx]
        zero_load_latency = float((avg_packet_length - 1) + hops * router_delay)
        path_congestion_delay = 0.0
        for hop_idx in range(hops):
            link_idx = path_matrix[src_idx, dst_idx, hop_idx]
            if link_idx == -1: continue
            link_load = link_loads_array[link_idx]
            arrival_rate = link_load / total_time_steps
            utilization = arrival_rate / link_bandwidth
            if utilization < 0.9999:
                wait_time = utilization / (1.0 - utilization)
                path_congestion_delay += avg_packet_length * wait_time
            else:
                path_congestion_delay += saturated_penalty
        latencies[i] = zero_load_latency + path_congestion_delay
    return latencies


class MacroGraphProcessor:

    def __init__(self, macro_graph: nx.DiGraph):
        print(f"--- [MacroGraphProcessor] Caching new graph (id: {id(macro_graph)}). Performing one-time pre-processing... ---")
        
        self.component_list = sorted(list(macro_graph.nodes()))
        self.comp_to_idx = {comp: i for i, comp in enumerate(self.component_list)}
        self.num_components = len(self.component_list)

        num_edges = macro_graph.number_of_edges()
        if num_edges == 0:
            self.src_indices = torch.tensor([], dtype=torch.long)
            self.dst_indices = torch.tensor([], dtype=torch.long)
            self.volumes_tensor = torch.tensor([], dtype=torch.float32)
        else:
            src_nodes = [None] * num_edges
            dst_nodes = [None] * num_edges
            weights = [None] * num_edges
            for i, (u, v, data) in enumerate(macro_graph.edges(data=True)):
                src_nodes[i] = u
                dst_nodes[i] = v
                weights[i] = data.get('weight', 0.0)

            src_indices_list = [self.comp_to_idx[s] for s in src_nodes]
            dst_indices_list = [self.comp_to_idx[d] for d in dst_nodes]

            self.src_indices = torch.tensor(src_indices_list, dtype=torch.long)
            self.dst_indices = torch.tensor(dst_indices_list, dtype=torch.long)
            self.volumes_tensor = torch.tensor(weights, dtype=torch.float32)

        print(f"--- Pre-processing for graph (id: {id(macro_graph)}) complete. ---")
        


class NoCUtils:

    _routing_matrix_cache: Dict[Tuple, Tuple[torch.Tensor, Dict[int, Tuple]]] = {}
    _distance_matrix_cache: Dict[Tuple, torch.Tensor] = {}
    _bisection_matrices_cache: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor]] = {}
    _path_matrices_cache: Dict[Tuple, Tuple[np.ndarray, np.ndarray]] = {} 
    
    @staticmethod
    def calculate_all_noc_metrics(
        activity_graph: nx.DiGraph,
        mapping_coords: Dict[str, Tuple[int, int]],
        noc_dims: Tuple[int, int],
        simulation_params: Optional[Dict] = None
    ) -> Dict[str, float]:

        if not mapping_coords:
            return {
                'total_energy_consumption': 0.0,
                'num_links_used': 0.0,
                'comm_cost': 0.0,
                'avg_packet_latency': 0.0,
                'avg_link_load': 0.0,
                'average_weighted_hops': 0.0
            }

        sim_params = simulation_params or {}
        final_metrics = {}

        comm_cost, total_inter_core_volume, avg_hops = NoCUtils.calculate_communication_metrics(
            activity_graph, mapping_coords, noc_dims
        )
        final_metrics['comm_cost'] = comm_cost
        final_metrics['average_weighted_hops'] = avg_hops

        link_loads_dict, _ = NoCUtils.calculate_link_loads(activity_graph, mapping_coords, noc_dims)
        congestion_metrics = NoCUtils.get_congestion_metrics(link_loads_dict)
        final_metrics['num_links_used'] = congestion_metrics['num_links_used']
        final_metrics['avg_link_load'] = congestion_metrics['avg_link_load']

        total_energy, _ = NoCUtils.calculate_energy_consumption(
            comm_cost, total_inter_core_volume, sim_params, activity_graph
        )
        final_metrics['total_energy_consumption'] = total_energy

        avg_latency, _ = NoCUtils.calculate_latency(activity_graph, mapping_coords, link_loads_dict, sim_params)
        final_metrics['avg_packet_latency'] = avg_latency

        return final_metrics
    
    @staticmethod
    def _get_path_matrices_numba(noc_dims: Tuple[int, int], link_to_idx: Dict[Tuple, int]) -> Tuple[np.ndarray, np.ndarray]:

        if noc_dims in NoCUtils._path_matrices_cache:
            return NoCUtils._path_matrices_cache[noc_dims]
        
        cache_dir = ".noc_cache"
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"path_matrices_{noc_dims[0]}x{noc_dims[1]}.npz")

        if os.path.exists(cache_file):
            try:
                print(f"      - [Cache Hit] Loading {noc_dims} path matrices from disk...")
                data = np.load(cache_file)
                path_matrix, path_lengths = data['path_matrix'], data['path_lengths']
                NoCUtils._path_matrices_cache[noc_dims] = (path_matrix, path_lengths)
                print("      - Path matrices loaded and cached in memory.")
                return path_matrix, path_lengths
            except Exception as e:
                print(f"      - [Cache Warning] Failed to load path matrix cache: {e}. Recalculating.")

        print(f"      - [First Call] Building path matrices for {noc_dims} NoC...")
        rows, cols = noc_dims
        num_cores = rows * cols
        core_coords = get_noc_core_coordinates(noc_dims)
        max_hops = (rows - 1) + (cols - 1)
        
        path_matrix = np.full((num_cores, num_cores, max_hops), -1, dtype=np.int32)
        path_lengths = np.zeros((num_cores, num_cores), dtype=np.int32)

        for i in range(num_cores):
            for j in range(num_cores):
                if i == j: continue
                
                path_coords = _get_xy_routing_path_numba(core_coords[i], core_coords[j])
                path_lengths[i, j] = len(path_coords) - 1
                
                for hop_idx in range(len(path_coords) - 1):
                    link = (path_coords[hop_idx], path_coords[hop_idx+1])
                    if link in link_to_idx:
                        path_matrix[i, j, hop_idx] = link_to_idx[link]
        
        NoCUtils._path_matrices_cache[noc_dims] = (path_matrix, path_lengths)
        try:
            np.savez_compressed(cache_file, path_matrix=path_matrix, path_lengths=path_lengths)
            print("      - Path matrices built and cached (memory and disk).")
        except Exception as e:
            print(f"      - [Cache Warning] Failed to save path matrices to disk: {e}.")

        return path_matrix, path_lengths

    @staticmethod
    def calculate_communication_metrics(
        activity_graph: nx.DiGraph,
        mapping_coords: Dict[str, Tuple[int, int]],
        noc_dims: Tuple[int, int],
        return_traffic_matrix: bool = False
    ):
        num_cores = noc_dims[0] * noc_dims[1]
        if not mapping_coords:
            return (0.0, 0.0, 0.0, torch.zeros((num_cores, num_cores))) if return_traffic_matrix else (0.0, 0.0, 0.0)

        core_coords_list = get_noc_core_coordinates(noc_dims)
        coord_to_idx = {coord: i for i, coord in enumerate(core_coords_list)}
        
        edge_info = _preprocess_graph_edges_for_comm(activity_graph, mapping_coords, coord_to_idx)
        
        traffic_matrix = torch.zeros((num_cores, num_cores), dtype=torch.float32)
        if edge_info:
            src_indices, dst_indices, volumes = edge_info
            if src_indices.numel() > 0:
                linear_indices = src_indices * num_cores + dst_indices
                traffic_matrix.view(-1).scatter_add_(0, linear_indices, volumes)
        
        distance_matrix = NoCUtils._get_distance_matrix_torch(noc_dims)
        comm_cost = torch.sum(traffic_matrix * distance_matrix).item()
        total_volume = torch.sum(traffic_matrix).item()
        avg_hops = comm_cost / (total_volume + 1e-9)

        if return_traffic_matrix:
            return comm_cost, total_volume, avg_hops, traffic_matrix
        else:
            return comm_cost, total_volume, avg_hops
    

    @staticmethod
    def calculate_link_loads(
        activity_graph: nx.DiGraph,
        mapping_coords: Dict[str, Tuple[int, int]],
        noc_dims: Tuple[int, int]
    ) -> Dict[Tuple[Tuple[int, int], Tuple[int, int]], float]:

        rows, cols = noc_dims
        num_cores = rows * cols
        if not mapping_coords:
            return NoCUtils._initialize_link_loads(noc_dims)

        traffic_matrix = torch.zeros((num_cores, num_cores), dtype=torch.float32)
        core_coords_list = get_noc_core_coordinates(noc_dims)
        coord_to_idx = {coord: i for i, coord in enumerate(core_coords_list)}

        for u, v, data in activity_graph.edges(data=True):
            if u in mapping_coords and v in mapping_coords:
                start_coord = mapping_coords[u]
                end_coord = mapping_coords[v]
                if start_coord != end_coord:
                    start_idx = coord_to_idx[start_coord]
                    end_idx = coord_to_idx[end_coord]
                    traffic_matrix[start_idx, end_idx] += float(data.get('source_activity', 0.0))

        routing_matrix, link_map = NoCUtils._get_routing_matrix_torch(noc_dims)
        num_links = routing_matrix.shape[2]
        
        traffic_vec = traffic_matrix.flatten().unsqueeze(0)

        routing_matrix_flat = routing_matrix.view(num_cores * num_cores, num_links)

        link_loads_vec = torch.matmul(traffic_vec, routing_matrix_flat).squeeze(0)
        
        link_loads_dict = {link: load.item() for link, load in zip(link_map.values(), link_loads_vec)}
        
        return link_loads_dict, traffic_matrix
    

    @staticmethod
    def get_congestion_metrics(link_loads: Dict) -> Dict[str, float]:

        if not link_loads:
            return {
                "max_link_load": 0.0, 
                "avg_link_load": 0.0, 
                "load_variance": 0.0, 
                "num_links_used": 0.0
            }
            
        load_values = np.array(list(link_loads.values()))
        
        return {
            "max_link_load": float(np.max(load_values)) if load_values.size > 0 else 0.0,
            "avg_link_load": float(np.mean(load_values)) if load_values.size > 0 else 0.0,
            "load_variance": float(np.var(load_values)) if load_values.size > 0 else 0.0,
            "num_links_used": float(np.sum(load_values > 1e-9))
        }
        

    @staticmethod
    def calculate_energy_consumption(
        communication_cost: float,
        total_inter_core_volume: float,
        simulation_params: Optional[Dict] = None,
        activity_graph: Optional[nx.DiGraph] = None
    ) -> Tuple[float, float]:
        
        sim_params = simulation_params or {}
        e_link = sim_params.get('energy_per_bit_link', 0.5)
        e_router = sim_params.get('energy_per_bit_router', 1.0)
        bits_per_flit = sim_params.get('bits_per_flit', 64)
        
        total_energy = (communication_cost * bits_per_flit * e_link) + \
                       (total_inter_core_volume * bits_per_flit * e_router)
        
        avg_energy = 0.0
        if activity_graph and activity_graph.number_of_nodes() > 0:
            avg_energy = total_energy / activity_graph.number_of_nodes()

        return total_energy, avg_energy
               

    @staticmethod
    def calculate_latency(
        activity_graph: nx.DiGraph,
        mapping_coords: Dict,
        link_loads: Dict,
        simulation_params: Optional[Dict] = None
    ) -> Tuple[float, float]:
        sim_params = simulation_params or {}
        total_time_steps = activity_graph.graph.get('total_time_steps', 1) * activity_graph.graph.get('num_samples', 1)
        if total_time_steps == 0: total_time_steps = 1

        if not mapping_coords: return 0.0, 0.0
        max_r = max(c[0] for c in mapping_coords.values())
        max_c = max(c[1] for c in mapping_coords.values())
        noc_dims = (max_r + 1, max_c + 1)
        
        coord_to_idx = {c: i for i, c in enumerate(get_noc_core_coordinates(noc_dims))}
        edge_info = _preprocess_graph_edges_for_comm(activity_graph, mapping_coords, coord_to_idx)
        
        if not edge_info or edge_info[0].numel() == 0:
            return 0.0, 0.0
        
        edges_info_np = torch.stack(edge_info, dim=1).numpy()
        
        _, link_map = NoCUtils._get_routing_matrix_torch(noc_dims)
        link_loads_array = np.array([link_loads.get(link_map[i], 0.0) for i in range(len(link_map))], dtype=np.float32)
        
        link_to_idx = {link: i for i, link in link_map.items()}
        path_matrix, path_lengths = NoCUtils._get_path_matrices_numba(noc_dims, link_to_idx)

        num_workers = os.cpu_count() //2 or 1
        chunk_size = max(1, len(edges_info_np) // (num_workers * 4))
        edge_chunks = [edges_info_np[i:i + chunk_size] for i in range(0, len(edges_info_np), chunk_size)]
        
        worker_args = (
            path_matrix, path_lengths, link_loads_array,
            sim_params.get('router_pipeline_delay', 2),
            sim_params.get('link_bandwidth', 1.0),
            sim_params.get('avg_packet_length', 5),
            1000.0 * sim_params.get('avg_packet_length', 5),
            total_time_steps
        )
        
        tasks = [delayed(_calculate_latency_worker_numba)(chunk, *worker_args) for chunk in edge_chunks]
        results_chunks = Parallel(n_jobs=num_workers)(tasks)
        
        if not results_chunks: return 0.0, 0.0
        all_packet_latencies = np.concatenate(results_chunks)
        
        volumes_np = edges_info_np[:, 2]
        total_volume = volumes_np.sum()
        if total_volume < 1e-9: return 0.0, 0.0

        avg_packet_latency = np.sum(all_packet_latencies * volumes_np) / total_volume
        max_packet_latency = np.max(all_packet_latencies)
        
        return float(avg_packet_latency), float(max_packet_latency)



    @staticmethod
    def calculate_saturation_throughput(
        activity_graph: nx.DiGraph,
        mapping_coords: Dict[str, Tuple[int, int]],
        noc_dims: Tuple[int, int],
        simulation_params: Optional[Dict] = None
    ) -> float:

        if not mapping_coords:
            return 0.0

        sim_params = simulation_params or {}
        link_bandwidth = sim_params.get('link_bandwidth', 1.0)
        num_samples = activity_graph.graph.get('num_samples')
        total_time_steps = activity_graph.graph.get('total_time_steps')

        if not (num_samples and total_time_steps):
            return 0.0
        
        total_simulation_time = num_samples * total_time_steps
        if total_simulation_time == 0:
            return 0.0
            
        rows, cols = noc_dims
        num_cores = rows * cols

        traffic_matrix = torch.zeros((num_cores, num_cores), dtype=torch.float32)
        core_coords_list = get_noc_core_coordinates(noc_dims)
        coord_to_idx = {coord: i for i, coord in enumerate(core_coords_list)}

        for u, v, data in activity_graph.edges(data=True):
            if u in mapping_coords and v in mapping_coords:
                start_coord = mapping_coords[u]
                end_coord = mapping_coords[v]
                start_idx = coord_to_idx[start_coord]
                end_idx = coord_to_idx[end_coord]
                traffic_matrix[start_idx, end_idx] += float(data.get('source_activity', 0.0))

        vertical_crossing_matrix, horizontal_crossing_matrix = NoCUtils._get_bisection_matrices_torch(noc_dims)

        vertical_bisection_load = torch.sum(traffic_matrix * vertical_crossing_matrix).item()
        
        horizontal_bisection_load = torch.sum(traffic_matrix * horizontal_crossing_matrix).item()

        vertical_bisection_arrival_rate = vertical_bisection_load / total_simulation_time
        horizontal_bisection_arrival_rate = horizontal_bisection_load / total_simulation_time
        
        vertical_bisection_bandwidth = rows * link_bandwidth
        horizontal_bisection_bandwidth = cols * link_bandwidth
        
        vertical_congestion_ratio = vertical_bisection_arrival_rate / (vertical_bisection_bandwidth + 1e-9)
        horizontal_congestion_ratio = horizontal_bisection_arrival_rate / (horizontal_bisection_bandwidth + 1e-9)
        
        max_congestion_ratio = max(vertical_congestion_ratio, horizontal_congestion_ratio)
        
        inter_core_traffic = traffic_matrix.clone()
        inter_core_traffic.fill_diagonal_(0)
        total_inter_core_volume = torch.sum(inter_core_traffic).item()
        total_average_rate = total_inter_core_volume / total_simulation_time

        if max_congestion_ratio > 1e-6:
            saturation_throughput = total_average_rate / max_congestion_ratio
        else:
            saturation_throughput = total_average_rate
            
        return saturation_throughput

    @staticmethod
    def calculate_average_energy_consumption(
        total_energy_consumption: float,
        activity_graph: nx.DiGraph
    ) -> float:
        num_neurons = activity_graph.number_of_nodes()
        if num_neurons == 0:
            return 0.0
        return total_energy_consumption / num_neurons

    
    @staticmethod
    def _initialize_link_loads(noc_dims: Tuple[int, int]) -> Dict[Tuple[Tuple[int, int], Tuple[int, int]], float]:

        rows, cols = noc_dims
        link_loads: Dict[Tuple[Tuple[int, int], Tuple[int, int]], float] = {}
        for r in range(rows):
            for c in range(cols):
                current_core_coord = (r, c)
                if c + 1 < cols:
                    right_core_coord = (r, c + 1)
                    link_loads[(current_core_coord, right_core_coord)] = 0.0
                    link_loads[(right_core_coord, current_core_coord)] = 0.0
                if r + 1 < rows:
                    down_core_coord = (r + 1, c)
                    link_loads[(current_core_coord, down_core_coord)] = 0.0
                    link_loads[(down_core_coord, current_core_coord)] = 0.0
        return link_loads

    @staticmethod
    def _get_xy_routing_path(start_coord, end_coord):
        path = [start_coord]
        curr_r, curr_c = start_coord
        end_r, end_c = end_coord
        while curr_c != end_c:
            curr_c += 1 if end_c > curr_c else -1
            path.append((curr_r, curr_c))
        while curr_r != end_r:
            curr_r += 1 if end_r > curr_r else -1
            path.append((curr_r, curr_c))
        return path
    
    
    @staticmethod
    def _get_distance_matrix_torch(noc_dims: Tuple[int, int]) -> torch.Tensor:
        if noc_dims in NoCUtils._distance_matrix_cache:
            return NoCUtils._distance_matrix_cache[noc_dims]
        
        core_coords = get_noc_core_coordinates(noc_dims)
        coords_tensor = torch.tensor(core_coords, dtype=torch.float32)
        distance_matrix = manhattan_distance_torch(coords_tensor, coords_tensor)
        
        NoCUtils._distance_matrix_cache[noc_dims] = distance_matrix
        return distance_matrix

    @staticmethod
    def _get_routing_matrix_torch(noc_dims: Tuple[int, int]) -> Tuple[torch.Tensor, Dict[int, Tuple]]:

        if noc_dims in NoCUtils._routing_matrix_cache:
            return NoCUtils._routing_matrix_cache[noc_dims]
        
        print(f"      - [First Call] Building routing matrix for {noc_dims} NoC in parallel...")
        rows, cols = noc_dims
        num_cores = rows * cols
        core_coords = get_noc_core_coordinates(noc_dims)

        link_map_rev = NoCUtils._initialize_link_loads(noc_dims).keys()
        link_map = {i: link for i, link in enumerate(link_map_rev)}
        num_links = len(link_map)
        link_to_idx = {link: i for i, link in link_map.items()}

        num_workers = os.cpu_count() // 2 or 1
        core_indices = list(range(num_cores))
        chunk_size = max(1, num_cores // num_workers)
        core_chunks = [core_indices[i:i + chunk_size] for i in range(0, num_cores, chunk_size)]

        tasks = [delayed(_build_routing_sub_matrix)(
            chunk, num_cores, core_coords, link_to_idx, num_links
        ) for chunk in core_chunks]

        sub_matrices = Parallel(n_jobs=num_workers)(tasks)
        
        routing_matrix = torch.cat(sub_matrices, dim=0)
        
        NoCUtils._routing_matrix_cache[noc_dims] = (routing_matrix, link_map)
        print("      - Routing matrix built and cached successfully.")
        return routing_matrix, link_map
    
    
    
    @staticmethod
    def _get_bisection_matrices_torch(noc_dims: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:

        if noc_dims in NoCUtils._bisection_matrices_cache:
            return NoCUtils._bisection_matrices_cache[noc_dims]
            
        rows, cols = noc_dims
        num_cores = rows * cols
        core_coords = get_noc_core_coordinates(noc_dims)
        
        bisection_col_idx = cols // 2
        core_cols = torch.tensor([c for _, c in core_coords], dtype=torch.int32)
        is_left = core_cols < bisection_col_idx
        is_right = core_cols >= bisection_col_idx
        
        vertical_crossing = (is_left.unsqueeze(1) & is_right.unsqueeze(0)) | \
                            (is_right.unsqueeze(1) & is_left.unsqueeze(0))

        bisection_row_idx = rows // 2
        core_rows = torch.tensor([r for r, _ in core_coords], dtype=torch.int32)
        same_col = core_cols.unsqueeze(1) == core_cols.unsqueeze(0)
        is_up = core_rows < bisection_row_idx
        is_down = core_rows >= bisection_row_idx
        
        horizontal_crossing = same_col & ((is_up.unsqueeze(1) & is_down.unsqueeze(0)) | \
                                          (is_down.unsqueeze(1) & is_up.unsqueeze(0)))
        
        result = (vertical_crossing.float(), horizontal_crossing.float())
        NoCUtils._bisection_matrices_cache[noc_dims] = result
        return result