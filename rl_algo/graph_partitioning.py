# -*- coding: utf-8 -*-

import argparse
from typing import List, Tuple, Union
import json
import os
import numpy as np

import networkx as nx
from rich import print

try:
    import sklearn
    import scipy
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from .HydroFlow_Mapping import run_hydroflow_mapping


__all__ = ['partition_graph']


def _hydroflow_wrapper(full_graph: nx.DiGraph, core_capacity: int, args: argparse.Namespace) -> Tuple[List[List[str]], np.ndarray]:

    num_cores = args.noc_rows * args.noc_cols
    core_capacities_np = None
    
    config_path = getattr(args, 'core_capacities_config', None)
    if config_path and os.path.exists(config_path):
        print(f"  [Partitioning Engine] [green]Loading heterogeneous core capacity config: {config_path}[/green]")
        try:
            with open(config_path, 'r') as f:
                capacities_list = json.load(f)
            core_capacities_np = np.array(capacities_list, dtype=np.int32)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse heterogeneous capacity config file '{config_path}': {e}")
    else:
        print(f"  [Partitioning Engine] [dim]No heterogeneous config provided, all core capacities will be uniformly set to: {core_capacity}[/dim]")
        core_capacities_np = np.full(num_cores, core_capacity, dtype=np.int32)

    return run_hydroflow_mapping(
        full_g=full_graph,
        c_caps=core_capacities_np,
        args=args
    )

ALGORITHM_REGISTRY = {
    "hydroflow_mapping": _hydroflow_wrapper,
}

FUNCTIONAL_ALGORITHMS = {
    'hydroflow_mapping'
}


def partition_graph(
    full_graph: nx.DiGraph,
    core_capacity: int,
    merge_threshold: int,
    args: argparse.Namespace
) -> Union[List[List[str]], Tuple[List[List[str]], np.ndarray]]:
    """
    
    """
    algorithm = getattr(args, 'partitioning_algorithm', 'hydroflow_mapping').lower()
    
    print(f"  [Partitioning Engine] Starting process, selected algorithm: [bold magenta]{algorithm.upper()}[/bold magenta]...")
    
    if not SKLEARN_AVAILABLE:
        raise ImportError("Graph partitioning or compilation algorithms require scikit-learn and scipy. Please install the relevant libraries.")
    
    if not full_graph or full_graph.number_of_nodes() == 0:
        print("  [Partitioning Engine] Input graph is empty, returning empty result.")
        return []

    handler_func = ALGORITHM_REGISTRY.get(algorithm)
    
    if handler_func is None:
        raise ValueError(f"Unknown graph partitioning or compilation algorithm: '{algorithm}'")

    if algorithm in FUNCTIONAL_ALGORITHMS:
        first_node_attrs = next(iter(full_graph.nodes(data=True)), (None, {}))[1]
        if 'freq_feature' not in first_node_attrs or not hasattr(first_node_attrs.get('freq_feature'), '__len__') or len(first_node_attrs['freq_feature']) == 0:
            if 'hydroflow' in algorithm or 'wavepart' in algorithm:
                print(f"  [yellow]Warning: Optional functional feature 'freq_feature' for algorithm '{algorithm}' not found in the graph. The functional affinity field will be disabled.[/yellow]")
            else:
                raise ValueError(f"Algorithm '{algorithm}' requires nodes to have a valid 'freq_feature' attribute, which was not found in the graph.")

    return handler_func(full_graph, core_capacity, args)