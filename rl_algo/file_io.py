# --- START OF FILE file_io.py ---

import os
import re
import sys
import argparse
import json
from typing import Any, Dict, List
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from rl_algo.noc_utils import manhattan_distance, NoCUtils
from rich import print

class FilePathManager:
    """
    
    """
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.snn_info: dict = {}
        self.analysis_dir: str = args.analysis_data_dir


    def update_snn_info(self, snn_info: dict):
        """
        
        """
        self.snn_info = snn_info
        arch_str = self.snn_info.get('model_arch_str', 'UnknownArch')
        t_inf = self.snn_info.get('inference_time_steps', -1)
        self.file_suffix = f"_{arch_str}_T{t_inf}"
        
        print(f"  [green]Path manager updated. File suffix to be used: [bold cyan]{self.file_suffix}[/bold cyan][/green]")
        os.makedirs(self.args.out_dir, exist_ok=True)
        
        
    def load_summary(self):
        """
        
        """
        print("\n--- Step 1: Loading SNN Analysis Summary File ---")
        summary_path = self._get_snn_summary_path()
        print(f"  Loading summary information from '{os.path.basename(summary_path)}'...")
        try:
            with open(summary_path, 'r', encoding='utf-8') as f:
                self.snn_info = json.load(f)
                self.update_snn_info(self.snn_info)
            print("  [green]SNN analysis summary loaded successfully.[/green]")
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"  [bold red]Error: Failed to load or parse SNN analysis summary file: {e}[/bold red]")
            sys.exit(1)

    def _get_snn_summary_path(self) -> str:
        """
        
        """
        if self.args.snn_summary_file and os.path.exists(self.args.snn_summary_file):
            return self.args.snn_summary_file
        
        if os.path.exists(self.analysis_dir):
            candidate_files = [f for f in os.listdir(self.analysis_dir) if f.startswith('snn_analysis_summary') and f.endswith('.json')]
            if candidate_files:
                candidate_files.sort(key=lambda fn: os.path.getmtime(os.path.join(self.analysis_dir, fn)), reverse=True)
                latest_file_path = os.path.join(self.analysis_dir, candidate_files[0])
                print(f"  [yellow]Auto-located the latest SNN analysis summary file:[/yellow] {latest_file_path}")
                return latest_file_path
        
        raise FileNotFoundError(f"Could not find `snn_analysis_summary...json` file in directory '{self.analysis_dir}'.")

    def get_path(self, file_key: str) -> str:
        """
        
        """
        if not self.snn_info:
            raise ValueError("Error: Must call load_summary() first to load the analysis summary.")
        
        filename = self.snn_info.get("generated_files", {}).get(file_key)
        if not filename:
            raise FileNotFoundError(f"Error: File key '{file_key}' not found in the analysis summary.")
            
        full_path = os.path.join(self.analysis_dir, filename)
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Error: File '{filename}' mentioned in the summary was not found at path '{full_path}'.")
            
        return full_path


def validate_input_files_for_graph_building(weights_path_no_ext: str, spikes_path: str):
    weights_full_path = f"{weights_path_no_ext}.npz"
    print(f"  Checking weights matrix: {weights_full_path}")
    if not os.path.exists(weights_full_path):
        print(f"Error: Weights matrix {weights_full_path} not found.")
        sys.exit(1)
    print(f"  Checking spike events: {spikes_path}")
    if not os.path.exists(spikes_path):
        print(f"Error: Spike events file {spikes_path} not found.")
        sys.exit(1)
    print("  [green]Weights and spike files exist.[/green]")

def load_activity_graph_from_gexf(gexf_file_path: str) -> nx.DiGraph:
    print(f"  Loading pre-computed activity graph from {gexf_file_path}...")
    if not os.path.exists(gexf_file_path):
        print(f"Error: Activity graph GEXF file {gexf_file_path} not found.")
        sys.exit(1)
    try:
        activity_graph = nx.read_gexf(gexf_file_path)
        for node, data in activity_graph.nodes(data=True):
            if 'total_spikes' in data: data['total_spikes'] = float(data.get('total_spikes', 0.0))
        for u, v, data in activity_graph.edges(data=True):
            if 'weight' in data: data['weight'] = float(data.get('weight', 0.0))
        print(f"  Successfully loaded activity graph: {activity_graph.number_of_nodes()} nodes, {activity_graph.number_of_edges()} edges.")
        if activity_graph.number_of_nodes() == 0:
            print("Error: Activity graph loaded from GEXF is empty.")
            sys.exit(1)
        return activity_graph
    except Exception as e:
        print(f"Error loading activity graph from GEXF file {gexf_file_path}: {e}")
        sys.exit(1)




def save_component_partitioning(output_path: str, components: List[Dict], activity_graph: nx.Graph, args: Any):
    """
    
    """
    print(f"  [Saving] Saving component partitioning results to: {output_path}")

    components_data_to_save = {}
    for comp in components:
        comp_id = comp['id']
        
        components_data_to_save[comp_id] = {
            "size": comp.get('size', 0),
            "internal_communication": comp.get('internal_comm', 0.0),
            "nodes": comp.get('nodes', []) 
        }

    output_data = {
        "metadata": {
            "source_graph_nodes": activity_graph.number_of_nodes(),
            "source_graph_edges": activity_graph.number_of_edges(),
            "partitioning_parameters": {
                "core_capacity": args.rl_core_capacity,
                "merge_threshold": args.component_merge_threshold
            },
            "total_components": len(components)
        },
        "components": components_data_to_save
    }

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4)
    except TypeError as e:
        print(f"[bold red]Error:[/bold red] Failed to save JSON, possibly because node IDs are not strings: {e}")
        for comp_data in components_data_to_save.values():
            comp_data['nodes'] = [str(n) for n in comp_data['nodes']]
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=4)
            print("  [Saving] Successfully saved after converting node IDs to strings.")
        except Exception as final_e:
            print(f"[bold red]Error:[/bold red] Still failed to save after converting to strings: {final_e}")
    except Exception as e:
        print(f"[bold red]Error:[/bold red] Failed to save component partitioning file: {e}")




def save_mapping_results(
    output_path: str, final_mapping_coords: dict, activity_graph: nx.DiGraph,
    noc_dims: tuple, snn_info: dict, args: argparse.Namespace,
    best_cost_metric: float, detailed_costs: dict
):
    """
    
    """
    print(f"  Saving mapping results to: {output_path}")
    
    if final_mapping_coords and detailed_costs:
        recalc_metrics = detailed_costs
    else:
        recalc_metrics = NoCUtils.calculate_all_noc_metrics(activity_graph, {}, noc_dims)

    primary_goal_arg = args.optimization_goal
    primary_goal_name = primary_goal_arg.replace('_', ' ').title()

    with open(output_path, "w", encoding="utf-8") as f_out:
        f_out.write(f"# NoC Mapping Results ({args.rl_agent_type.upper()})\n")
        f_out.write(f"# SNN Architecture: {snn_info.get('model_arch_str', 'Unknown')}\n")
        f_out.write(f"# T={snn_info.get('inference_time_steps', 'Unknown')}, Analyzed Samples={snn_info.get('num_samples_analyzed', 'Unknown')}\n")
        f_out.write(f"# Activity Graph Source Model: {snn_info.get('source_model_path', 'Unknown')}\n")
        f_out.write(f"# Activity Graph Nodes: {activity_graph.number_of_nodes()}\n")
        f_out.write(f"# Activity Graph Edges: {activity_graph.number_of_edges()}\n")
        f_out.write(f"# Activity Graph Source: {args.graph_source}, Build Strategy={args.graph_build_strategy}\n")
        f_out.write(f"# RL Immediate Reward Strategy: {args.immediate_reward_strategy}\n")
        f_out.write(f"# NoC Dimensions: {noc_dims[0]}x{noc_dims[1]}, Core Capacity: {args.rl_core_capacity}\n")
        f_out.write("#\n")
        f_out.write(f"# ----- Evaluation of Best Solution Found During Training -----\n")
        f_out.write(f"# Primary Optimization Goal: {primary_goal_name}\n")
        f_out.write(f"# Best Goal Value: {best_cost_metric:,.4f}\n")
        f_out.write(f"# Composite Score: {recalc_metrics.get('composite_score', float('inf')):,.4f}\n")
        f_out.write("#\n")
        f_out.write(f"# ----- Details of All Performance Metrics -----\n")
        
        for key, value in sorted(recalc_metrics.items()):
            if isinstance(value, (int, float)):
                if key != 'composite_score':
                    f_out.write(f"# - {key.replace('_', ' ').title()}: {value:,.4f}\n")
                  
                    
        f_out.write("\n# ----- All Parameters -----\n")
        for key, value in vars(args).items():
            f_out.write(f"# {key}: {value}\n")
        
            
    print(f"  Mapping results saved. Best {primary_goal_name}: {best_cost_metric:,.2f}")