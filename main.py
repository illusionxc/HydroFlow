# --- START OF FILE main.py ---
import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rl_algo.rl_mapper_engine import RLMapperEngine

def main():
    parser = argparse.ArgumentParser(
        description='General Reinforcement Learning NoC Mapper Execution Script',
        formatter_class=argparse.RawTextHelpFormatter
    )

    # --- File and SNN Information Parameters ---
    file_group = parser.add_argument_group('File and SNN Information Parameters')
    file_group.add_argument(
        '--analysis_data_dir', type=str,
        default="result/snn_analysis_output/deepsnn_layers256_256_128_128_64_64_32_32_16_16_tau2.0_T8_lr0.0001_e10_cifar10",
        help='Root directory of SNN analysis artifacts, containing all relevant output files.'
    )
    file_group.add_argument(
        '--snn_summary_file', type=str,
        default='result/snn_analysis_output/deepsnn_layers256_256_128_128_64_64_32_32_16_16_tau2.0_T8_lr0.0001_e10_cifar10/snn_analysis_summary_deepsnn_tau2.0_cifar10_T100.json',
        help='Full path to the SNN analysis summary JSON file (e.g., "output/summary.json").'
    )
    parser.add_argument(
        '--summary_filepath', type=str, 
        default='result/snn_analysis_output/deepsnn_layers256_256_128_128_64_64_32_32_16_16_tau2.0_T8_lr0.0001_e10_cifar10/snn_analysis_summary_deepsnn_tau2.0_cifar10_T100.json',
        help='Full path to the SNN analysis summary JSON file (e.g., "output/summary.json").'
    )
    parser.add_argument(
        '--spike_events_filepath', type=str,
        default='result/snn_analysis_output/deepsnn_layers256_256_128_128_64_64_32_32_16_16_tau2.0_T8_lr0.0001_e10_cifar10/spike_events_deepsnn_tau2.0_cifar10_T100.txt',
        help='Full path to the SNN spike events TXT file (e.g., "output/spike_events.txt").'
    )
    parser.add_argument('--peak_window_size', default=10, type=int, help='Peak joint activity window size')
    parser.add_argument('--response_delay_min', default=0, type=int, help='Minimum response delay')
    parser.add_argument('--response_delay_max', default=1, type=int, help='Maximum response delay')
    parser.add_argument('--use_static_weights_in_joint_strength', default=False, type=bool,
                        help='Multiply by absolute static weights in joint activity strength')

    # --- Activity Graph Source and Build Strategy ---
    graph_group = parser.add_argument_group('Activity Graph Source and Build Strategy')
    graph_group.add_argument(
        '--graph_source', type=str, default='build_online', choices=['load_gexf', 'build_online'],
        help='Source of the activity graph:\n'
             '- "load_gexf": Load from a pre-generated GEXF file.\n'
             '- "build_online": (Recommended) Build the graph in memory at runtime.'
    )
    graph_group.add_argument(
        '--online_build_mode', type=str, default='peak_static_full_topo',)
    graph_group.add_argument(
        '--activity_graph_gexf_file', type=str,
        default='result/snn_analysis_output/deepsnn_layers128_64_64_32_tau2.0_T8_lr0.001_e1_cifar10/peak_joint_activity_graph_deepsnn_layers128646432_tau2.0_cifar10_T100_strategy_static_full_topo_peak_delay0-1.gexf')
 
    # --- NoC Parameters ---
    noc_group = parser.add_argument_group('NoC Parameters')
    noc_group.add_argument('--noc_rows', type=int, default=8, help='Number of rows in the NoC grid.')
    noc_group.add_argument('--noc_cols', type=int, default=8, help='Number of columns in the NoC grid.')
    noc_group.add_argument('--core_capacity', type=int, default=32, help='Maximum number of neurons per NoC core.')
    noc_group.add_argument('--rl_core_capacity', type=int, default=32, help='Maximum number of neurons per NoC core, reference value: 10~50')
    noc_group.add_argument(
        '--core_capacities_config', type=str, default=None,
        help="[Heterogeneous] (Optional) Path to a JSON file defining heterogeneous core capacities: [cap_core0, cap_core1, ...]."
    )
    
    # --- NoC Simulation Hardware Parameters ---
    sim_group = parser.add_argument_group('NoC Simulation Hardware Parameters')
    sim_group.add_argument('--link_bandwidth', type=float, default=1.0, help='Bandwidth of NoC links (flits/cycle).')
    sim_group.add_argument('--router_pipeline_delay', type=int, default=2, help='Pipeline delay of a single router (cycles).')
    sim_group.add_argument('--bits_per_flit', type=int, default=64, help='Number of bits per flit.')
    sim_group.add_argument('--avg_packet_length', type=int, default=5, help='Average packet length in flits for latency models.')
    sim_group.add_argument('--energy_per_bit_link', type=float, default=0.5, help='Energy per bit for link traversal (pJ/bit).')
    sim_group.add_argument('--energy_per_bit_router', type=float, default=1.0, help='Energy per bit for router processing (pJ/bit).')
    sim_group.add_argument('--burst_window_size', type=int, default=10,help='Sliding window size for analyzing bursty congestion, in time_steps. Suggested: 10 ~ 50')

    # --- SNN Feature Extractor Parameters ---
    feature_extractor_params = parser.add_argument_group('SNN Spatio-Temporal Feature Extractor Parameters')
    feature_extractor_params.add_argument('--gcpn_use_freq_features', type=bool, default=True, help="Enable calculation of frequency-domain features for neurons.")
    feature_extractor_params.add_argument('--gcpn_freq_feat_dim', type=int, default=10)
    feature_extractor_params.add_argument('--snn_sampling_rate', type=float, default=1000.0, help="Sampling rate of SNN spike data (Hz).")
    feature_extractor_params.add_argument('--feat_inc_fft', type=bool, default=True, help="Extract FFT-based features.")
    feature_extractor_params.add_argument('--fft_pooling_method', type=str, default='binning', help="FFT feature pooling method.")
    feature_extractor_params.add_argument('--feat_inc_wavelet', type=bool, default=True, help="Extract wavelet-based features.")
    feature_extractor_params.add_argument('--wavelet_family', type=str, default='db4', help="Wavelet family for analysis (e.g., 'db4', 'sym8').")
    feature_extractor_params.add_argument('--wavelet_levels', type=int, default=2, help="Number of wavelet decomposition levels.")
    feature_extractor_params.add_argument('--feat_inc_hurst', type=bool, default=True, help="Extract the Hurst exponent feature.")
    feature_extractor_params.add_argument('--feat_inc_graph_spectrum', type=bool, default=False, help="Extract graph spectral features.")
    feature_extractor_params.add_argument('--feat_inc_coherence', type=bool, default=True, help="Extract edge-level cross-spectral coherence.")
    feature_extractor_params.add_argument('--coherence_nperseg', type=int, default=256, help="Segment length for Welch's method in coherence calculation.")

    # --- Hierarchical Mapping & Partitioning Parameters ---
    hierarchical_group = parser.add_argument_group('Hierarchical Mapping & Partitioning Parameters')
    hierarchical_group.add_argument(
        '--mapping_mode', type=str, default='hierarchical')
    hierarchical_group.add_argument(
        '--partitioning_algorithm', type=str, default='hydroflow_mapping',
        help="Algorithm for graph partitioning or integrated mapping."
    )
    hierarchical_group.add_argument(
        '--component_merge_threshold', type=int, default=0,
        help="Threshold for merging small components after partitioning."
    )
    
    # --- HydroFlow-Mapping Hyperparameters ---
    hydroflow_group = parser.add_argument_group('HydroFlow-Mapping Hyperparameters')
    hydroflow_group.add_argument('--visualize_hydroflow', type=bool, default=False, help="Enable visualization animations during HydroFlow mapping.")
    hydroflow_group.add_argument('--w_topo', type=float, default=1.0, help='[HydroFlow] Topological interaction force weight.')
    hydroflow_group.add_argument('--w_func', type=float, default=1.0, help='[HydroFlow] Functional affinity weight.')
    hydroflow_group.add_argument('--w_repel', type=float, default=0.1, help='[HydroFlow] Base weight for core load repulsion force.')
    hydroflow_group.add_argument('--w_comm', type=float, default=1e-5, help="[HydroFlow] Communication load repulsion force weight.")
    hydroflow_group.add_argument('--w_io', type=float, default=0.1, help="[HydroFlow] I/O center attraction force weight.")
    hydroflow_group.add_argument('--hf_temperature_initial', type=float, default=100.0, help="[HydroFlow] [Stochastic] Initial temperature for simulated annealing.")
    hydroflow_group.add_argument('--hf_temperature_cooling_factor', type=float, default=0.9, help="[HydroFlow] [Stochastic] Cooling factor for simulated annealing.")
    hydroflow_group.add_argument('--hf_enable_pruning', type=bool, default=True, help="[HydroFlow] Enable candidate core pruning optimization.")
    hydroflow_group.add_argument('--hf_pruning_radius', type=int, default=3, help="[HydroFlow] Search radius for candidate core pruning.")
    

    # --- Optimization Goal and Reward Configuration ---
    opt_group = parser.add_argument_group('Optimization Goal and Reward Configuration')
    opt_group.add_argument('--optimization_goal', type=str, default='comm_cost')
    opt_group.add_argument('--metric_w_comm_cost', type=float, default=1.0e-3, help='[Minimize] Weight for Communication Cost.')
    opt_group.add_argument('--metric_w_max_link_load', type=float, default=0.0, help='[Minimize] Weight for Maximum Link Load.')
    opt_group.add_argument('--metric_w_load_variance', type=float, default=0.0, help='[Minimize] Weight for Link Load Variance.')
    opt_group.add_argument('--metric_w_avg_packet_latency', type=float, default=50.0, help='[Minimize] Weight for Average Packet Latency.')
    opt_group.add_argument('--metric_w_max_packet_latency', type=float, default=0.0, help='[Minimize] Weight for Maximum Packet Latency.')
    opt_group.add_argument('--metric_w_total_energy_consumption', type=float, default=0.0, help='[Minimize] Weight for Total Energy Consumption.')
    opt_group.add_argument('--metric_w_throughput', type=float, default=0.0, help='[Maximize] Weight (negative) for Total Throughput.')
    opt_group.add_argument('--metric_w_saturation_throughput', type=float, default=0.0, help='[Maximize] Weight (negative) for Saturation Throughput.')
    opt_group.add_argument('--metric_w_average_weighted_hops', type=float, default=0.0, help='[Minimize] Weight for Average Weighted Hops.')
    opt_group.add_argument(
        '--k_neighbors_reward', type=int, default=5,
        help='The number of neighbors K to consider when using the "k_neighbors" strategy.'
    )
    opt_group.add_argument('--immediate_reward_strategy', type=str, default='k_neighbors')

    # --- Other Parameters ---
    other_group = parser.add_argument_group('Other Parameters')
    other_group.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='Computation device.')
    other_group.add_argument('--out_dir', type=str, default='result/rl_mapper_output', help='Root directory for output results.')
    other_group.add_argument('--enable-memory-profiling', action='store_true', help='Enable memory analysis tool (requires memory-profiler).')
    
    other_group.add_argument(
        '--rl_agent_type', type=str, 
        default='test',
        
        help='Select the Reinforcement Learning algorithm to use.'
    )
    
    other_group.add_argument(
        '--graph_build_strategy', type=str, 
        default='static_full_topo',
    )
    
    args = parser.parse_args()
    
    try:
        engine = RLMapperEngine(args)
        engine.run()
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
