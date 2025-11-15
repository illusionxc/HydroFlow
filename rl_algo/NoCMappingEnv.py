# --- START OF FILE NoCMappingEnv.py ---

import torch
import networkx as nx
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple
from rl_algo.noc_utils import manhattan_distance, get_noc_core_coordinates, NoCUtils
from rich import print
from collections import defaultdict
import torch.jit as jit

@jit.script
def _calculate_state_jit(
    current_neuron_activity: float,
    max_neuron_activity: float,
    core_loads: torch.Tensor,
    core_capacity: float,
    is_mapping_empty: bool,
    comm_weights: torch.Tensor,
    placed_cores_indices: torch.Tensor,
    core_coords_tensor: torch.Tensor,
    max_possible_comm_cost_per_step: float,
    num_cores: int
) -> torch.Tensor:
    
    norm_activity = torch.tensor([current_neuron_activity / (max_neuron_activity + 1e-9)], device=core_loads.device)

    is_available_mask = core_loads < core_capacity
    norm_loads = core_loads / core_capacity
    
    est_comm_cost = torch.zeros(num_cores, device=core_loads.device)
    if not is_mapping_empty:
        placed_core_coords = core_coords_tensor[placed_cores_indices]
        dist_tensor = torch.abs(core_coords_tensor.unsqueeze(1) - placed_core_coords.unsqueeze(0))
        manhattan_distances = torch.sum(dist_tensor, dim=2)
        est_comm_cost = torch.matmul(manhattan_distances, comm_weights.unsqueeze(1)).squeeze(1)

    est_comm_cost.masked_fill_(~is_available_mask, 0.0)
    
    norm_est_comm = torch.clamp_max(est_comm_cost / (max_possible_comm_cost_per_step + 1e-9), 5.0)

    core_features = torch.stack([
        is_available_mask.float(),
        norm_loads,
        norm_est_comm
    ], dim=1).flatten()

    state_tensor = torch.cat([norm_activity, core_features]).unsqueeze(0)
    return state_tensor



class NoCMappingEnv:

    def __init__(self,
                 activity_graph: nx.DiGraph,
                 neurons_to_place_ordered: List[str],
                 noc_dims: Tuple[int, int],
                 device: torch.device,
                 core_capacity: int,
                 reward_config: Dict,
                 max_neuron_activity: Optional[float] = None,
                 core_coords_list: Optional[List[Tuple[int, int]]] = None,
                 core_assignments_template: Optional[Dict[int, List[str]]] = None,
                 core_load_template: Optional[List[int]] = None):

        self.activity_graph = activity_graph
        self.neurons_to_place_ordered = neurons_to_place_ordered
        self.num_total_neurons_to_place = len(neurons_to_place_ordered)
        self.noc_dims = noc_dims
        self.num_cores = int(noc_dims[0] * noc_dims[1])
        self.device = device
        self.core_capacity = core_capacity
        
        if core_coords_list is not None:
            self.core_coords_list = core_coords_list
        else:
            self.core_coords_list = get_noc_core_coordinates(noc_dims)

        self.core_coords_tensor = torch.tensor(self.core_coords_list, dtype=torch.float32, device=self.device)
        
        assert reward_config is not None, "reward_config 字典必须被提供"
        self.reward_config = reward_config
        self.immediate_reward_strategy_name = self.reward_config.get('immediate_reward_strategy', 'k_neighbors')
        self.k_neighbors = self.reward_config['k_neighbors_reward']

        if max_neuron_activity is not None:
            self.max_neuron_activity = max_neuron_activity
        else:
            node_activities = [float(data.get('total_spikes', 0.0)) for _, data in self.activity_graph.nodes(data=True)]
            self.max_neuron_activity = max(node_activities) if node_activities and max(node_activities) > 0 else 1.0
        
        max_dist_val = (self.noc_dims[0] - 1) + (self.noc_dims[1] - 1) if self.num_cores > 1 else 0
        self.max_possible_comm_cost_per_step = self.max_neuron_activity * max_dist_val * self.core_capacity

        self.current_neuron_idx_in_order: int = 0
        self.mapping_neuron_to_core_idx: Dict[str, int] = {}
        
        if core_assignments_template is not None:
            self.core_assignments = core_assignments_template.copy()
        else:
            self.core_assignments: Dict[int, List[str]] = {i: [] for i in range(self.num_cores)}

        if core_load_template is not None:
            self.core_current_load: List[int] = list(core_load_template)
        else:
            self.core_current_load: List[int] = [0] * self.num_cores
        
        self.reset()
    
        
        
    def reset(self) -> torch.Tensor:

        self.current_neuron_idx_in_order = 0
        self.mapping_neuron_to_core_idx.clear()
        
        self.core_assignments = {i: [] for i in range(self.num_cores)}
        
        self.core_current_load = [0] * self.num_cores
        
        return self._calculate_state()
    

    def step(self, action_core_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:

        current_neuron_id = self._get_current_neuron_id()
        
        if current_neuron_id is None:
            return (self._calculate_state(),
                    torch.tensor([[0.0]], device=self.device, dtype=torch.float32),
                    torch.tensor([[True]], device=self.device, dtype=torch.bool),
                    {"status": "All neurons placed"})

        if not self._is_core_available_for_placement(action_core_idx):
            immediate_comm_cost = self.reward_config['invalid_action_penalty']
            done = False
            info = {"error": "Invalid action: core capacity exceeded."}
            
            next_state = self._calculate_state()
            return (next_state,
                    torch.tensor([[immediate_comm_cost]], device=self.device, dtype=torch.float32),
                    torch.tensor([[done]], device=self.device, dtype=torch.bool),
                    info)

        self.mapping_neuron_to_core_idx[current_neuron_id] = action_core_idx
        self.core_assignments[action_core_idx].append(current_neuron_id)
        self.core_current_load[action_core_idx] += 1
        
        
        immediate_reward = 0.0
        info = {}

        if not self.disable_immediate_reward:
            cost = self.immediate_reward_func(self, current_neuron_id, action_core_idx)
            
            immediate_reward = -cost * self.reward_config['rw_immediate_comm_weight']
            info["immediate_cost_step"] = cost
            info["immediate_reward_step"] = immediate_reward


        self.current_neuron_idx_in_order += 1
        done = self.current_neuron_idx_in_order >= self.num_total_neurons_to_place
        
        next_state = self._calculate_state()
        
        return (next_state, torch.tensor([[immediate_reward]], device=self.device, dtype=torch.float32), 
                torch.tensor([[done]], device=self.device, dtype=torch.bool), info)


    def _get_current_neuron_id(self) -> Optional[str]:

        if self.current_neuron_idx_in_order < self.num_total_neurons_to_place:
            return self.neurons_to_place_ordered[self.current_neuron_idx_in_order]
        return None

    def _is_core_available_for_placement(self, core_idx: int) -> bool:

        return self.core_current_load[core_idx] < self.core_capacity

    def get_available_actions_mask(self) -> List[bool]:

        return [self._is_core_available_for_placement(i) for i in range(self.num_cores)]


    def _calculate_state(self) -> torch.Tensor:

        current_neuron_id = self._get_current_neuron_id()
        if current_neuron_id is None:
            return torch.zeros(self.get_state_size(), device=self.device).unsqueeze(0)

        current_neuron_data = self.activity_graph.nodes.get(current_neuron_id, {})
        current_neuron_activity = float(current_neuron_data.get('total_spikes', 0.0))
        
        core_loads = torch.tensor(self.core_current_load, device=self.device, dtype=torch.float32)
        
        is_mapping_empty = not self.mapping_neuron_to_core_idx
        comm_weights = torch.empty(0, device=self.device, dtype=torch.float32)
        placed_cores_indices = torch.empty(0, device=self.device, dtype=torch.long)

        if not is_mapping_empty:
            placed_neurons = list(self.mapping_neuron_to_core_idx.keys())
            placed_cores_indices = torch.tensor(list(self.mapping_neuron_to_core_idx.values()), device=self.device, dtype=torch.long)
            
            comm_weights_list = [
                self.activity_graph.get_edge_data(current_neuron_id, nid, {}).get('source_activity', 0.0) +
                self.activity_graph.get_edge_data(nid, current_neuron_id, {}).get('source_activity', 0.0)
                for nid in placed_neurons
            ]
            comm_weights = torch.tensor(comm_weights_list, device=self.device, dtype=torch.float32)

        return _calculate_state_jit(
            current_neuron_activity,
            self.max_neuron_activity,
            core_loads,
            float(self.core_capacity),
            is_mapping_empty,
            comm_weights,
            placed_cores_indices,
            self.core_coords_tensor,
            self.max_possible_comm_cost_per_step,
            self.num_cores
        )
        
        

    def get_state_size(self) -> int:

        return 1 + self.num_cores * 3
    
    
    def get_state_snapshot(self) -> Tuple[Dict[str, int], Optional[str], None]:

        return (
            self.mapping_neuron_to_core_idx.copy(), 
            self._get_current_neuron_id(),
            None
        )
        
        
    def get_core_occupancy_grid(self) -> np.ndarray:

        grid = np.zeros((self.noc_dims[0], self.noc_dims[1]), dtype=int)
        for core_idx, neurons in self.core_assignments.items():
            if neurons:
                r, c = self.core_coords_list[core_idx]
                grid[r, c] = 1
        return grid