# --- START OF FILE utils.py ---

# -*- coding: utf-8 -*-
"""
"""
import os
from collections import defaultdict
from typing import List, Dict, Any, Tuple
import networkx as nx
import numpy as np
from rich import print

def build_macro_graph(full_graph: nx.DiGraph, components: Dict[str, Any]) -> nx.DiGraph:
    """
    """
    macro_graph = nx.DiGraph()
    node_to_comp_map = {node: comp_id for comp_id, comp_data in components.items() for node in comp_data['nodes']}
    
    comp_total_external_weight = defaultdict(float)
    for u, v, data in full_graph.edges(data=True):
        comp_u = node_to_comp_map.get(u)
        comp_v = node_to_comp_map.get(v)
        if comp_u and comp_v and comp_u != comp_v:
            weight = data.get('source_activity', data.get('weight', 0.0))
            comp_total_external_weight[comp_u] += weight
            comp_total_external_weight[comp_v] += weight

    for comp_id, comp_data in components.items():
        subgraph = full_graph.subgraph(comp_data['nodes'])
        internal_comm = subgraph.size(weight='source_activity')
        macro_graph.add_node(
            comp_id, 
            size=len(comp_data['nodes']), 
            internal_comm=internal_comm,
            nodes=list(comp_data['nodes']),
            total_external_weight=comp_total_external_weight.get(comp_id, 0.0)
        )

    for u, v, data in full_graph.edges(data=True):
        comp_u = node_to_comp_map.get(u)
        comp_v = node_to_comp_map.get(v)
        if comp_u and comp_v and comp_u != comp_v:
            weight = data.get('source_activity', data.get('weight', 0.0))
            if weight > 0:
                if macro_graph.has_edge(comp_u, comp_v):
                    macro_graph[comp_u][comp_v]['weight'] += weight
                else:
                    macro_graph.add_edge(comp_u, comp_v, weight=weight)
    return macro_graph