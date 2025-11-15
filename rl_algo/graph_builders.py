import random
from typing import Any, Dict, List, Optional, Tuple, Set
from collections import defaultdict
from matplotlib import pyplot as plt
import networkx as nx
import torch
import os
import sys
import argparse
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
import scipy
import tqdm
from rich import print
from networkx.algorithms import community
try:
    from networkx.algorithms import community
    COMMUNITY_DETECTION_AVAILABLE = True
except ImportError:
    COMMUNITY_DETECTION_AVAILABLE = False

try:
    import powerlaw
    POWERLAW_AVAILABLE = True
except ImportError:
    POWERLAW_AVAILABLE = False



import networkx as nx
import numpy as np
import os
from typing import Dict, Any, Optional, Tuple
import tqdm


def _get_layer_by_name(topology: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
    
    
    
    for layer in topology.get('layers', []):
        if layer.get('name') == name: return layer
    return None


def _get_neuron_shape_after_post_op(layer_info: Dict[str, Any]) -> Tuple[int, ...]:
    
    
    
    shape = tuple(layer_info.get('neuron_shape', ()))
    post_op = layer_info.get('post_op')
    if not post_op or not shape: return shape

    if post_op.get('type') == 'MaxPool2d' and len(shape) >= 3:
        h_in, w_in, stride = shape[1], shape[2], post_op.get('params', {}).get('stride', 2)
        stride_h, stride_w = (stride, stride) if isinstance(stride, int) else stride
        h_out, w_out = h_in // stride_h, w_in // stride_w
        return (shape[0], h_out, w_out)
    
    if post_op.get('type') == 'AdaptiveAvgPool2d_Flatten': return (shape[0],)
    
    return shape