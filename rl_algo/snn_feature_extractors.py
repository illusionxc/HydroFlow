
import argparse
import time
import numpy as np
import networkx as nx
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
from rich import print
import os
from scipy.signal import welch, csd
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import MiniBatchKMeans

try:
    import pywt
    PYWAVELET_AVAILABLE = True
except ImportError:
    PYWAVELET_AVAILABLE = False
    print("[yellow]Warning: `pywavelets` library not installed, wavelet feature extraction will be unavailable.[/yellow]")

try:
    from scipy.signal import coherence
    SCIPY_SIGNAL_AVAILABLE = True
except ImportError:
    SCIPY_SIGNAL_AVAILABLE = False
    print("[yellow]Warning: `scipy` version is too low or incomplete, cross-spectral coherence feature extraction may be unavailable.[/yellow]")

try:
    from hurst import compute_Hc
    HURST_AVAILABLE = True
except ImportError:
    HURST_AVAILABLE = False
    print("[yellow]Warning: `hurst` library not installed, Hurst exponent feature extraction will be unavailable.[/yellow]")


def _preprocess_node_signal_worker(
    node_id_str: Any, 
    spike_times: List[int], 
    total_time_steps: int
) -> Tuple[Any, Optional[np.ndarray]]:
    
    
    
    if not spike_times:
        return node_id_str, None
        
    signal = np.zeros(total_time_steps, dtype=np.int8)
    valid_times = [t for t in spike_times if 0 <= t < total_time_steps]
    if not valid_times:
        return node_id_str, None
        
    signal[valid_times] = 1
    
    return node_id_str, signal if np.sum(signal) > 0 else None


def _safe_coherence_worker(
    signal1: np.ndarray, 
    signal2: np.ndarray, 
    fs: float, 
    nperseg: int
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:

    try:
        freqs, pxx = welch(signal1, fs=fs, nperseg=nperseg)
        _, pyy = welch(signal2, fs=fs, nperseg=nperseg)
        
        _, pxy = csd(signal1, signal2, fs=fs, nperseg=nperseg)
        
        cxy = np.zeros_like(freqs)
        denominator = pxx * pyy
        
        valid_indices = np.where(denominator > 1e-12)[0]
        
        if len(valid_indices) > 0:
            cxy[valid_indices] = np.abs(pxy[valid_indices])**2 / denominator[valid_indices]
            
        return freqs, cxy
        
    except Exception:
        return None, None
    
    

def _process_single_node_fft_from_signal(
    node_id_str: Any, 
    signal: np.ndarray,
    sampling_rate: float, 
    num_features: int, 
    pooling_method: str, 
    freq_bins: Dict
) -> Tuple[Any, np.ndarray]:
    
    spectrum = np.fft.rfft(signal)
    power_spectrum = np.abs(spectrum)**2
    norm = np.linalg.norm(power_spectrum)
    normalized_spectrum = power_spectrum / (norm + 1e-9)

    if pooling_method == 'binning':
        freqs = np.fft.rfftfreq(len(signal), d=1.0 / sampling_rate)
        binned_features = [np.sum(normalized_spectrum[np.where((freqs >= low) & (freqs < high))]) for _, (low, high) in freq_bins.items()]
        return node_id_str, np.array(binned_features, dtype=np.float32)
    else: # 'none'
        feat_dim = min(len(normalized_spectrum), num_features)
        final_feat = np.zeros(num_features, dtype=np.float32)
        final_feat[:feat_dim] = normalized_spectrum[:feat_dim]
        return node_id_str, final_feat


def _process_single_node_wavelet_from_signal(
    node_id_str: Any,
    signal: np.ndarray,
    wavelet_family: str,
    num_levels: int,
    feature_names: List[str]
) -> Tuple[Any, Optional[np.ndarray]]:
    
    if np.sum(signal) < 2: 
        return node_id_str, None
        
    try:
        wavelet_obj = pywt.Wavelet(wavelet_family)
        max_level = pywt.dwt_max_level(len(signal), wavelet_obj)
        effective_level = min(num_levels, max_level)
        if effective_level <= 0: return node_id_str, None
        coeffs = pywt.wavedec(signal, wavelet=wavelet_obj, level=effective_level, mode='symmetric')
    except ValueError:
        return node_id_str, None

    level_features = []
    for c in coeffs:
        energy = np.sum(c**2)
        c_prob = c**2 / (np.sum(c**2) + 1e-9)
        entropy = -np.sum(c_prob * np.log2(c_prob + 1e-9))
        level_features.extend([energy, entropy])
    
    padding_needed = len(feature_names) - len(level_features)
    if padding_needed > 0: level_features.extend([0.0] * padding_needed)
    
    avg_features = np.array(level_features, dtype=np.float32)
    norm = np.linalg.norm(avg_features)
    return node_id_str, avg_features / (norm + 1e-9)


def _process_single_node_hurst_from_signal(
    node_id_str: Any,
    signal: np.ndarray,
) -> Tuple[Any, Optional[np.ndarray]]:
    
    if len(np.unique(signal)) < 2 or len(signal) < 20:
        return node_id_str, None
        
    try:
        H, _, _ = compute_Hc(signal, kind='random_walk', simplified=True)
        return node_id_str, np.array([np.clip(H, 0, 1)], dtype=np.float32)
    except Exception:
        return node_id_str, None


class SNNFeatureExtractor:
    
    
    

    @staticmethod
    def extract_all_features(
        graph: nx.Graph,
        detailed_spike_data: Dict,
        args: argparse.Namespace
    ) -> Dict[str, Any]:
        
        
        
        print("\n[bold magenta]========== SNN Advanced Spatio-Temporal Feature Extraction Started (V5 Performance Edition) ==========[/bold magenta]")
        start_time = time.time()
        
        params = vars(args)
        snn_node_list = list(graph.nodes())
        
        print("\n[bold cyan]--- Step 1/5: Unified Data Preprocessing (Parallelized) ---[/bold cyan]")
        preprocessed_signals = SNNFeatureExtractor._preprocess_signals(
            snn_node_list, detailed_spike_data
        )
        if not preprocessed_signals:
            print("[red]Error: Data preprocessing failed to generate any valid signals, feature extraction aborted.[/red]")
            return {'node_features': {}, 'edge_features': {}, 'node_feature_names': [], 'edge_feature_names': []}
        
        print("\n[bold cyan]--- Step 2/5: Extracting Basic Node Features ---[/bold cyan]")
        node_feature_tasks_defs = []
        if params.get('feat_inc_fft', True):
            node_feature_tasks_defs.append({
                "func": SNNFeatureExtractor._extract_fft_features, "name": "FFT",
                "default": lambda dim: np.zeros(dim),
                "args": [snn_node_list, preprocessed_signals, params['snn_sampling_rate'], params['gcpn_freq_feat_dim'], params['fft_pooling_method']]
            })
        if params.get('feat_inc_wavelet', False):
            node_feature_tasks_defs.append({
                "func": SNNFeatureExtractor._extract_wavelet_features, "name": "Wavelet",
                "default": lambda dim: np.zeros(dim),
                "args": [snn_node_list, preprocessed_signals, params['wavelet_family'], params['wavelet_levels']]
            })
        if params.get('feat_inc_hurst', False):
            node_feature_tasks_defs.append({
                "func": SNNFeatureExtractor._extract_hurst_features, "name": "Hurst",
                "default": lambda dim: np.full(dim, 0.5),
                "args": [snn_node_list, preprocessed_signals]
            })

        all_node_features_parts = {}
        for task_def in node_feature_tasks_defs:
            features, names = task_def['func'](*task_def['args'])
            all_node_features_parts[task_def['name']] = (features, names)
        
        final_node_features, final_node_feature_names = {}, []
        for task_def in node_feature_tasks_defs:
            _, names = all_node_features_parts[task_def['name']]
            final_node_feature_names.extend(names)

        for node_id in snn_node_list:
            feature_vector_parts = []
            for task_def in node_feature_tasks_defs:
                features, names = all_node_features_parts[task_def['name']]
                if not names: continue
                default_val = task_def['default'](len(names))
                feature_part = features.get(node_id, default_val)
                feature_vector_parts.append(feature_part)
            
            if feature_vector_parts:
                final_node_features[node_id] = np.concatenate(feature_vector_parts)
        
        print(f"[bold green]Basic node feature extraction complete. Dimension: {len(final_node_feature_names)}[/bold green]")

        print("\n[bold cyan]--- Step 3/5: Extracting Dependency Features (Graph Spectrum) ---[/bold cyan]")
        if params.get('feat_inc_graph_spectrum', False):
            if final_node_feature_names:
                nx.set_node_attributes(graph, final_node_features, 'freq_feature')
                dependent_features, dependent_names = SNNFeatureExtractor._extract_graph_spectral_features(
                    graph, snn_node_list, params.get('graph_spec_k', 10), params.get('graph_spec_radius', 10)
                )
                if dependent_names:
                    final_node_feature_names.extend(dependent_names)
                    default_dep = np.zeros(len(dependent_names))
                    for node_id in snn_node_list:
                        dep_feat = dependent_features.get(node_id, default_dep)
                        final_node_features[node_id] = np.concatenate([final_node_features.get(node_id, []), dep_feat])
                    print(f"[bold green]Dependency feature extraction complete. Final total feature dimension: {len(final_node_feature_names)}[/bold green]")
            else:
                print("[red]Error -> Skipping: Cannot compute graph spectral features because no basic features were successfully computed.[/red]")

        print("\n[bold cyan]--- Step 4/5: Extracting Edge Features (Approximate Coherence) ---[/bold cyan]")
        edge_features, edge_feature_names = {}, []
        if params.get('feat_inc_coherence', True):
            total_time_steps = detailed_spike_data.get('total_time_steps')
            if total_time_steps:
                edge_features, edge_feature_names = SNNFeatureExtractor._calculate_cross_spectral_features_approx(
                    graph, preprocessed_signals, total_time_steps,
                    params.get('snn_sampling_rate', 1000.0),
                    params.get('coherence_nperseg', 256),
                    params.get('coherence_num_clusters', 32)
                )
        
        print("\n[bold cyan]--- Step 5/5: Integrating Final Results ---[/bold cyan]")
        end_time = time.time()
        print(f"  Total feature extraction time: {end_time - start_time:.4f} seconds")
        
        print("\n[bold magenta]========== SNN Advanced Spatio-Temporal Feature Extraction Complete ==========[/bold magenta]")
        
        return {
            'node_features': final_node_features,
            'edge_features': edge_features,
            'node_feature_names': final_node_feature_names,
            'edge_feature_names': edge_feature_names
        }

    @staticmethod
    def _run_parallel(tasks: List, desc: str) -> Dict:
        
        
        
        num_workers = os.cpu_count() or 1
        print(f"    [Parallel] Using {num_workers} CPU cores for parallel computation of {desc} features...")
        
        results = Parallel(n_jobs=num_workers, pre_dispatch='2*n_jobs')(
            tqdm(tasks, desc=f"      Calculating {desc}", ncols=100)
        )

        return {key: val for key, val in results if val is not None}

    @staticmethod
    def _preprocess_signals(snn_node_list: List[Any], detailed_spike_data: Dict) -> Dict[Any, np.ndarray]:
        
        
        
        spike_times_per_sample = detailed_spike_data.get('spike_times_per_sample', {})
        num_samples = detailed_spike_data.get('num_samples', 1)
        total_time_steps = detailed_spike_data.get('total_time_steps')
        if not all([spike_times_per_sample, total_time_steps]): return {}
        
        print("  - [Preprocessing] Aggregating spike data from all samples...")
        aggregated_spikes = defaultdict(list)
        for sample_id in range(num_samples):
            for neuron_id, times in spike_times_per_sample.get(sample_id, {}).items():
                aggregated_spikes[neuron_id].extend(times)
        
        print("  - [Preprocessing] Parallel reconstruction of 0-1 signals...")
        tasks = []
        for node_id_str in snn_node_list:
            try:
                node_id_int = int(str(node_id_str).replace('neuron_', '').replace('comp_', ''))
                tasks.append(delayed(_preprocess_node_signal_worker)(
                    node_id_str, aggregated_spikes.get(node_id_int, []), total_time_steps
                ))
            except (ValueError, AttributeError):
                continue
        
        preprocessed_signals = SNNFeatureExtractor._run_parallel(tasks, "Signals")
        print(f"  - [Preprocessing] Complete, signals generated for {len(preprocessed_signals)} active nodes.")
        return preprocessed_signals

    @staticmethod
    def _extract_fft_features(snn_node_list: List[Any], preprocessed_signals: Dict, sampling_rate: float, num_features: int, pooling_method: str) -> Tuple[Dict, List]:
        
        feature_names, freq_bins = [], {}
        if pooling_method == 'binning':
            freq_bins = {'fft_delta': (0.5, 4), 'fft_theta': (4, 8), 'fft_alpha': (8, 13), 'fft_beta': (13, 30), 'fft_gamma': (30, 100)}
            feature_names = list(freq_bins.keys())
        else: # 'none'
            feature_names = [f'fft_low_freq_{i}' for i in range(num_features)]
        
        tasks = [delayed(_process_single_node_fft_from_signal)(node_id, signal, sampling_rate, num_features, pooling_method, freq_bins)
                 for node_id, signal in preprocessed_signals.items()]
        
        return SNNFeatureExtractor._run_parallel(tasks, "FFT"), feature_names

    @staticmethod
    def _extract_wavelet_features(snn_node_list: List[Any], preprocessed_signals: Dict, wavelet_family: str, num_levels: int) -> Tuple[Dict, List]:
        
        if not PYWAVELET_AVAILABLE: return {}, []
        feature_names = []
        feature_names.extend([f"wavelet_cA{num_levels}_energy", f"wavelet_cA{num_levels}_entropy"])
        for i in range(num_levels, 0, -1):
            feature_names.extend([f"wavelet_cD{i}_energy", f"wavelet_cD{i}_entropy"])

        tasks = [delayed(_process_single_node_wavelet_from_signal)(node_id, signal, wavelet_family, num_levels, feature_names)
                 for node_id, signal in preprocessed_signals.items()]
        
        return SNNFeatureExtractor._run_parallel(tasks, "Wavelet"), feature_names

    @staticmethod
    def _extract_hurst_features(snn_node_list: List[Any], preprocessed_signals: Dict) -> Tuple[Dict, List]:
        
        if not HURST_AVAILABLE: return {}, []
        tasks = [delayed(_process_single_node_hurst_from_signal)(node_id, signal)
                 for node_id, signal in preprocessed_signals.items()]
        return SNNFeatureExtractor._run_parallel(tasks, "Hurst"), ['hurst_H']

    @staticmethod
    def _extract_graph_spectral_features(graph: nx.Graph, snn_node_list: List[Any], k: int, radius: int) -> Tuple[Dict, List]:
        
        feature_names = ['spectral_cosine_similarity', 'num_neighbors']
        default_feature = np.array([0.5, 0.0], dtype=np.float32)
        try:
            all_freq_features = {node: data['freq_feature'] for node, data in graph.nodes(data=True) if 'freq_feature' in data and data['freq_feature'].size > 0}
        except KeyError:
            print("  [red]Error: Node is missing 'freq_feature' attribute, cannot compute local spectral difference.[/red]")
            return {node_id: default_feature for node_id in snn_node_list}, feature_names
        
        if not all_freq_features:
            return {node_id: default_feature for node_id in snn_node_list}, feature_names
        
        undirected_view = nx.to_undirected(graph)
        neuron_features = {}
        for node_id in tqdm(snn_node_list, desc="    Calculating Spectral Locality", ncols=100):
            feat_u = all_freq_features.get(node_id)
            if feat_u is None:
                neuron_features[node_id] = default_feature
                continue
            
            neighbors = list(nx.neighbors(undirected_view, node_id))
            neighbor_feats = [all_freq_features.get(n) for n in neighbors if all_freq_features.get(n) is not None]
            
            if not neighbor_feats:
                neuron_features[node_id] = np.array([0.5, float(len(neighbors))], dtype=np.float32)
                continue
            
            feat_neighbors_avg = np.mean(neighbor_feats, axis=0)
            similarity = cosine_similarity(feat_u.reshape(1, -1), feat_neighbors_avg.reshape(1, -1))[0, 0]
            neuron_features[node_id] = np.array([similarity, float(len(neighbors))], dtype=np.float32)
        return neuron_features, feature_names


    @staticmethod
    def _calculate_cross_spectral_features_approx(
        graph: nx.Graph,
        preprocessed_signals: Dict[Any, np.ndarray],
        total_time_steps: int,
        fs: float,
        nperseg: int,
        n_clusters: int
    ) -> Tuple[Dict[Tuple[Any, Any], Dict[str, float]], List[str]]:
        
        
        
        if not SCIPY_SIGNAL_AVAILABLE or not preprocessed_signals:
            return {}, []
        
        print("  - [Coherence Approx] Calculating edge-level cross-spectral coherence using cluster approximation...")
        
        node_ids_with_signal = list(preprocessed_signals.keys())
        signal_matrix = np.array([preprocessed_signals[nid] for nid in node_ids_with_signal], dtype=np.float32)

        print(f"    [1/3] Performing K-Means clustering on {len(node_ids_with_signal)} signals (K={n_clusters})...")
        actual_n_clusters = min(n_clusters, len(node_ids_with_signal))
        if actual_n_clusters <= 1:
            print("    [yellow]Insufficient number of signals for meaningful clustering, skipping coherence calculation.[/yellow]")
            return {}, []
            
        kmeans = MiniBatchKMeans(n_clusters=actual_n_clusters, random_state=42, n_init='auto', batch_size=256)
        labels = kmeans.fit_predict(signal_matrix)
        node_to_cluster_label = {node_id: label for node_id, label in zip(node_ids_with_signal, labels)}

        print("    [2/3] Parallel computation of cross-spectral coherence between cluster centers (warning-free mode)...")
        cluster_centers = kmeans.cluster_centers_
        coherence_matrix = np.zeros((actual_n_clusters, actual_n_clusters))
        
        safe_nperseg = min(nperseg, total_time_steps)
        tasks = [delayed(_safe_coherence_worker)(cluster_centers[i], cluster_centers[j], fs, safe_nperseg)
                 for i in range(actual_n_clusters) for j in range(i, actual_n_clusters)]
        
        results = Parallel(n_jobs=-1)(tqdm(tasks, desc="      Calculating center coherence", ncols=100))

        task_idx = 0
        for i in range(actual_n_clusters):
            for j in range(i, actual_n_clusters):
                _, Cxy = results[task_idx]
                max_coh = np.max(Cxy) if (Cxy is not None and len(Cxy) > 0) else 0.0
                coherence_matrix[i, j] = max_coh
                coherence_matrix[j, i] = max_coh
                task_idx += 1
        
        print("    [3/3] Assigning approximate coherence values to all edges...")
        final_edge_features = {}
        for u, v in tqdm(graph.edges(), desc="      Assigning coherence", ncols=100):
            label_u = node_to_cluster_label.get(u)
            label_v = node_to_cluster_label.get(v)
            if label_u is not None and label_v is not None:
                approx_coherence = coherence_matrix[label_u, label_v]
                final_edge_features[(u, v)] = {'coherence': float(approx_coherence)}
        
        print(f"    [green]Complete. Approximate coherence assigned to {len(final_edge_features)} edges.[/green]")
        return final_edge_features, ['coherence']

    
    @staticmethod
    def get_feature_names(args: argparse.Namespace) -> Dict[str, List[str]]:
        
        
        
        node_names, edge_names = [], []

        if getattr(args, 'feat_inc_fft', False):
            if getattr(args, 'fft_pooling_method', 'binning') == 'binning':
                node_names.extend(['fft_delta', 'fft_theta', 'fft_alpha', 'fft_beta', 'fft_gamma'])
            else:
                node_names.extend([f'fft_low_freq_{i}' for i in range(getattr(args, 'gcpn_freq_feat_dim', 10))])
        
        if getattr(args, 'feat_inc_wavelet', False):
            levels = getattr(args, 'wavelet_levels', 2)
            wavelet_names = [f"wavelet_cA{levels}_energy", f"wavelet_cA{levels}_entropy"]
            for i in range(levels, 0, -1):
                wavelet_names.extend([f"wavelet_cD{i}_energy", f"wavelet_cD{i}_entropy"])
            node_names.extend(wavelet_names)
        
        if getattr(args, 'feat_inc_hurst', False):
            node_names.append('hurst_H')
            
        if getattr(args, 'feat_inc_graph_spectrum', False):
            node_names.extend(['spectral_cosine_similarity', 'num_neighbors'])

        if getattr(args, 'feat_inc_coherence', False):
            edge_names.append('coherence')
        
        return {
            "node_feature_names": node_names,
            "edge_feature_names": edge_names
        }