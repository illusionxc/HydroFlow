import os, time, gc
import networkx as nx
import numpy as np
from rich import print
import numba
from rl_algo.noc_utils import get_noc_core_coordinates

EPS = 1e-9

@numba.njit('float32(float32[:], float32[:])', fastmath=True, cache=True)
def cos_sim(u, v):
    uv, su, sv = np.float32(0.0), np.float32(0.0), np.float32(0.0)
    for i in range(u.shape[0]):
        uv += u[i] * v[i]; su += u[i] * u[i]; sv += v[i] * v[i]
    n_u, n_v = np.sqrt(su), np.sqrt(sv)
    if n_u < EPS or n_v < EPS: return np.float32(0.0)
    return uv / (n_u * n_v)

@numba.njit(fastmath=True, cache=True)
def calc_aff_pot(v_feat, cand_c_ids, c_centroids, c_sizes):
    n_cands = len(cand_c_ids)
    aff_pots = np.zeros(n_cands, dtype=np.float32)
    for i in range(n_cands):
        c_id = cand_c_ids[i]
        if c_sizes[c_id] > 0:
            aff_pots[i] = cos_sim(v_feat, c_centroids[c_id])
    return aff_pots

@numba.njit(fastmath=True, cache=True)
def softmax_t(pots, temp):
    if temp < EPS:
        probs = np.zeros_like(pots); probs[np.argmax(pots)] = 1.0; return probs
    
    pots = pots - np.max(pots)
    exp_pots = np.exp(pots / temp)
    sum_exp = np.sum(exp_pots)
    
    if sum_exp < EPS: return np.full_like(pots, 1.0 / len(pots))
    return exp_pots / sum_exp

@numba.njit(fastmath=True, cache=True)
def hf_kernel(
    temp_init, temp_cool_f, do_prune, prune_rad, do_analysis, analysis_iv, total_w,
    v_order, adj_ind, adj_l, adj_w, feat_mat, n_acts, n_degs, n_cores, c_caps,
    has_feats, w_inter, w_aff, w_load_base, w_comm, w_io,
    c_coords, ctr_coord, max_dist, c_dist_mat
):
    n_nodes, n_feats = len(v_order), feat_mat.shape[1]
    
    n2c_map = np.full(n_nodes, -1, dtype=np.int32)
    c_sizes = np.zeros(n_cores, dtype=np.int32)
    c_centroids = np.zeros((n_cores, n_feats), dtype=np.float32)
    c_traffic = np.zeros(n_cores, dtype=np.float32)
    curr_temp = temp_init
    
    num_samples = (n_nodes // analysis_iv) if do_analysis and analysis_iv > 0 else 0
    analysis_hist = np.zeros((num_samples, 3), dtype=np.float32)
    s_idx, intra_w = 0, np.float32(0.0)
    
    visited = np.zeros(n_nodes, dtype=np.bool_)

    dyn_w_thresh, dyn_w_exp, lr_decay = np.float32(0.5), np.float32(2.0), np.float32(0.005)

    for i in range(n_nodes):
        v = v_order[i]
        
        dyn_w_load = w_load_base
        ratios = c_sizes / (c_caps + EPS)
        max_ratio = np.max(ratios)
        if max_ratio > dyn_w_thresh:
            p_factor = (max_ratio - dyn_w_thresh) / (1.0 - dyn_w_thresh + EPS)
            enhancement = 1.0 + (np.exp(p_factor * dyn_w_exp) - 1.0)
            dyn_w_load *= enhancement

        cand_cores = np.zeros(n_cores, dtype=np.int32)
        n_cands = 0
        if do_prune:
            neighbor_com, total_n_w = np.zeros(2, dtype=np.float32), np.float32(0.0)
            start, end = adj_ind[v], adj_ind[v + 1]
            for ptr in range(start, end):
                n_id = adj_l[ptr]
                m_core = n2c_map[n_id]
                if m_core != -1:
                    w = adj_w[ptr]
                    neighbor_com += c_coords[m_core] * w
                    total_n_w += w
            if total_n_w > EPS:
                neighbor_com /= total_n_w
                for c_id in range(n_cores):
                    if c_sizes[c_id] < c_caps[c_id]:
                        d_com = np.sum(np.abs(c_coords[c_id] - neighbor_com))
                        if d_com <= prune_rad:
                            cand_cores[n_cands] = c_id; n_cands += 1
        
        if n_cands == 0:
            n_cands = 0
            for c_id in range(n_cores):
                if c_sizes[c_id] < c_caps[c_id]: cand_cores[n_cands] = c_id; n_cands += 1

        if n_cands == 0: 
            best_c = np.argmin(c_sizes)
        else:
            cand_ids = cand_cores[:n_cands]
            
            inter_pot = np.zeros(n_cands, dtype=np.float32)
            v_start, v_end = adj_ind[v], adj_ind[v + 1]
            
            map_n_cores, map_n_weights = [], []
            for ptr in range(v_start, v_end):
                n_id = adj_l[ptr]; visited[n_id] = True
                m_core = n2c_map[n_id]
                if m_core != -1: map_n_cores.append(m_core); map_n_weights.append(adj_w[ptr])
            
            if len(map_n_cores) > 0:
                m_n_cores = np.array(map_n_cores, dtype=np.int32)
                m_n_weights = np.array(map_n_weights, dtype=np.float32)
                dists = c_dist_mat[cand_ids, :][:, m_n_cores]
                inter_pot = np.sum((max_dist - dists) * m_n_weights, axis=1)

            for n1_ptr in range(v_start, v_end):
                n1 = adj_l[n1_ptr]
                n1_start, n1_end = adj_ind[n1], adj_ind[n1 + 1]
                for n2_ptr in range(n1_start, n1_end):
                    n2 = adj_l[n2_ptr]
                    if n2 != v and not visited[n2]:
                        m_core = n2c_map[n2]
                        if m_core != -1:
                            dists_to_n2 = c_dist_mat[cand_ids, m_core]
                            inter_pot += lr_decay * adj_w[n2_ptr] * (max_dist - dists_to_n2)
            for ptr in range(v_start, v_end): visited[adj_l[ptr]] = False

            aff_pot = np.zeros(n_cands, dtype=np.float32)
            if has_feats: aff_pot = calc_aff_pot(feat_mat[v], cand_ids, c_centroids, c_sizes)

            load_cost = c_sizes[cand_ids].astype(np.float32)
            comm_cost = c_traffic[cand_ids]
            d_ctr = np.sum(np.abs(c_coords[cand_ids] - ctr_coord), axis=1)
            io_pot = n_degs[v] * (max_dist - d_ctr)

            pots = (w_inter * inter_pot) + (w_aff * aff_pot) - (dyn_w_load * load_cost) - (w_comm * comm_cost) + (w_io * io_pot)
            
            probs = softmax_t(pots, curr_temp)
            choice_idx = np.searchsorted(np.cumsum(probs), np.random.random(), side="right")
            best_c = cand_ids[choice_idx]

        n2c_map[v] = best_c
        
        if has_feats:
            old_s = c_sizes[best_c]
            new_cent = (c_centroids[best_c] * old_s + feat_mat[v]) / (old_s + 1.0)
            c_centroids[best_c] = new_cent
        c_sizes[best_c] += 1
        c_traffic[best_c] += n_acts[v]
        
        if do_analysis:
            start, end = adj_ind[v], adj_ind[v+1]
            for n_ptr in range(start, end):
                neighbor = adj_l[n_ptr]
                if n2c_map[neighbor] == best_c and v < neighbor:
                    intra_w += adj_w[n_ptr]
        
        curr_temp *= temp_cool_f 

        if do_analysis and (i + 1) % analysis_iv == 0 and s_idx < num_samples:
            phi_s = intra_w / total_w if total_w > EPS else 0.0
            phi_f = 0.0
            if has_feats:
                n_non_empty = 0
                for c_idx in range(n_cores):
                    if c_sizes[c_idx] > 0: phi_f += np.linalg.norm(c_centroids[c_idx]); n_non_empty += 1
                if n_non_empty > 0: phi_f /= n_non_empty
            
            analysis_hist[s_idx] = [i + 1, phi_s, phi_f]; s_idx += 1
            
    return n2c_map, analysis_hist

class HFPreproc:
    @staticmethod
    @numba.njit(cache=True)
    def agg_edges(us, vs, ws):
        if len(us) == 0: return (np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32))
        
        out_e = []
        curr_u, curr_v, curr_w = us[0], vs[0], ws[0]
        
        for i in range(1, len(us)):
            u, v, w = us[i], vs[i], ws[i]
            if u == curr_u and v == curr_v: curr_w += w
            else:
                out_e.append((curr_u, curr_v, curr_w))
                curr_u, curr_v, curr_w = u, v, w
        out_e.append((curr_u, curr_v, curr_w))
        
        num_out = len(out_e)
        u_out, v_out, w_out = np.empty(num_out,dtype=np.int32), np.empty(num_out,dtype=np.int32), np.empty(num_out,dtype=np.float32)
        for i in range(num_out): u_out[i], v_out[i], w_out[i] = out_e[i]
        return u_out, v_out, w_out

    @staticmethod
    @numba.njit(cache=True)
    def build_csr(u_c, v_c, w_c, n_nodes):
        degs = np.zeros(n_nodes, dtype=np.int32)
        for i in range(len(u_c)): degs[u_c[i]] += 1; degs[v_c[i]] += 1
        adj_ind = np.zeros(n_nodes + 1, dtype=np.int32)
        for i in range(n_nodes): adj_ind[i+1] = adj_ind[i] + degs[i]
        adj_l = np.zeros(adj_ind[-1], dtype=np.int32)
        adj_w = np.zeros(adj_ind[-1], dtype=np.float32)
        curr_pos = np.zeros(n_nodes, dtype=np.int32)
        for i in range(len(u_c)):
            u, v, w = u_c[i], v_c[i], w_c[i]
            pos_u = adj_ind[u] + curr_pos[u]; adj_l[pos_u] = v; adj_w[pos_u] = w; curr_pos[u] += 1
            pos_v = adj_ind[v] + curr_pos[v]; adj_l[pos_v] = u; adj_w[pos_v] = w; curr_pos[v] += 1
        return adj_ind, adj_l, adj_w, degs

    @staticmethod
    def graph_to_csr(g, n2i):
        cpl_srcs, cpl_ws = ['joint_activity_strength', 'coherence'], [1, 1]
        
        n_nodes, edges = len(n2i), list(g.edges(data=True))
        if not edges: return (np.zeros(n_nodes + 1, dtype=np.int32), np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32), np.zeros(n_nodes, dtype=np.int32), 0.0)
        
        sum_w = sum(cpl_ws)
        if sum_w > EPS: cpl_ws = [w / sum_w for w in cpl_ws]

        uvw_list = []
        for u, v, data in edges:
            if u not in n2i or v not in n2i: continue
            src_vals = [float(data.get(src, 0.0)) for src in cpl_srcs]
            j_eff = sum(val * w for val, w in zip(src_vals, cpl_ws))
            if j_eff < EPS: j_eff = float(data.get('weight', 0.0))
            if j_eff > 0: uvw_list.append((n2i[u], n2i[v], j_eff))

        if not uvw_list: return (np.zeros(n_nodes + 1, dtype=np.int32), np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32), np.zeros(n_nodes, dtype=np.int32), 0.0)

        e_arr = np.array(uvw_list, dtype=np.float32)
        us, vs, ws = e_arr[:, 0].astype(np.int32), e_arr[:, 1].astype(np.int32), e_arr[:, 2]
        total_w = np.sum(ws)
        
        mask = us > vs
        us[mask], vs[mask] = vs[mask], us[mask]
        sort_idx = np.lexsort((vs, us))
        us, vs, ws = us[sort_idx], vs[sort_idx], ws[sort_idx]
        
        u_c, v_c, w_c = HFPreproc.agg_edges(us, vs, ws)
        adj_ind, adj_l, adj_w, degs = HFPreproc.build_csr(u_c, v_c, w_c, n_nodes)
        return adj_ind, adj_l, adj_w, degs, total_w
    
    @staticmethod
    def get_n_acts(full_g, n2i, act_src):
        n_nodes = len(n2i)
        acts = np.zeros(n_nodes, dtype=np.float32)
        fb_order = ['burstiness', 'isi_cv', 'source_activity', 'total_spikes', 0.0]
        pref_order = [act_src] + [item for item in fb_order if item != act_src]
        
        for node, data in full_g.nodes(data=True):
            act_val = None
            for key in pref_order:
                if isinstance(key, str) and key in data: act_val = data[key]; break
            if act_val is None: act_val = pref_order[-1]
            acts[n2i[node]] = float(act_val)
        return acts

def run_hydroflow_mapping(full_g, c_caps, args):
    print("\n[bold magenta]========== [HydroFlow-Mapping V6.0] Firing up the fluid sim... ==========[/bold magenta]")

    w_inter = np.float32(getattr(args, 'w_topo', 1.0))
    w_aff = np.float32(getattr(args, 'w_func', 1.0))
    w_load_base = np.float32(getattr(args, 'w_repel', 0.1))
    w_comm = np.float32(getattr(args, 'w_comm', 1e-5))
    w_io = np.float32(getattr(args, 'w_io', 0.1))
    t_init = np.float32(getattr(args, 'hf_temperature_initial', 100.0))
    t_cool = np.float32(getattr(args, 'hf_temperature_cooling_factor', 0.95))
    do_prune = bool(getattr(args, 'hf_enable_pruning', True))
    prune_rad = int(getattr(args, 'hf_pruning_radius', 3))
    noc_dims = (args.noc_rows, args.noc_cols)
    n_cores = noc_dims[0] * noc_dims[1]
    
    if c_caps.shape[0] != n_cores: raise ValueError(f"Bad core cap length {c_caps.shape[0]}, expected {n_cores}")
    
    print("    Pre-calculating core distances...")
    c_coords = np.array(get_noc_core_coordinates(noc_dims), dtype=np.int32)
    c_dist_mat = np.zeros((n_cores, n_cores), dtype=np.int32)
    for i in range(n_cores):
        for j in range(i, n_cores):
            dist = np.sum(np.abs(c_coords[i] - c_coords[j])); c_dist_mat[i,j]=dist; c_dist_mat[j,i]=dist
    max_dist = (noc_dims[0] - 1) + (noc_dims[1] - 1)
    ctr_coord = np.array([(noc_dims[0] - 1) / 2.0, (noc_dims[1] - 1) / 2.0], dtype=np.float32)
    
    f_len = next((len(d['freq_feature']) for _,d in full_g.nodes(data=True) if 'freq_feature' in d and d['freq_feature'] is not None),0)
    has_feats = f_len > 0
    
    print("    Sim config: Stochastic, Dynamic Load, Pruning ON. Weights: T{:.1f}/F{:.1f}/L{:.2f}/C{:.1e}/IO{:.1f}".format(w_inter, w_aff, w_load_base, w_comm, w_io))

    t_start_total = time.time()
    
    print("\n[bold]>>> [1/3: Pre-proc] Building CSR & Features...[/bold]")
    t_pre = time.time()
    if full_g.number_of_nodes() == 0: return [], np.array([]), np.array([]), {}
    
    og_nodes, n2i = list(full_g.nodes()), {node: i for i, node in enumerate(list(full_g.nodes()))}
    
    adj_ind, adj_l, adj_w, degs, total_w = HFPreproc.graph_to_csr(full_g, n2i)
    
    feat_mat = np.zeros((len(og_nodes), max(1, f_len)), dtype=np.float32)
    if has_feats:
        for node_str, data in full_g.nodes(data=True):
            if 'freq_feature' in data and data['freq_feature'] is not None and len(data['freq_feature'])==f_len:
                feat_mat[n2i[node_str]] = data['freq_feature']
    
    n_acts = HFPreproc.get_n_acts(full_g, n2i, 'burstiness')
    print(f"      [green]Done. ({time.time() - t_pre:.3f}s)[/green]")

    print("\n[bold]>>> [2/3: Relaxation] Running kernel...[/bold]")
    t_sim = time.time()

    w_degs = np.zeros(len(og_nodes), dtype=np.float32)
    for i in range(len(adj_ind) - 1): w_degs[i] = np.sum(adj_w[adj_ind[i]:adj_ind[i+1]])
    v_order = np.argsort(w_degs)[::-1].copy()

    n2c_map_np, analysis_hist = hf_kernel(
        t_init, t_cool, do_prune, prune_rad,
        True, 10, total_w,
        v_order, adj_ind, adj_l, adj_w, feat_mat, n_acts, degs, n_cores, c_caps, 
        has_feats, w_inter, w_aff, w_load_base, w_comm, w_io, c_coords, ctr_coord, max_dist, c_dist_mat
    )
    
    print(f"      [green]System relaxed. ({time.time() - t_sim:.3f}s)[/green]")

    print("\n[bold]>>> [3/3: Finalizing] Formatting results...[/bold]")
    t_fin = time.time()
    del adj_ind, adj_l, adj_w, feat_mat, degs, n_acts; gc.collect() # memory conscious
    
    i2n = {i: node for node, i in n2i.items()}
    
    final_parts = [[] for _ in range(n_cores)]
    for node_int, c_id in enumerate(n2c_map_np):
        if c_id != -1: final_parts[c_id].append(i2n[node_int])
    
    print(f"      [green]Done. ({time.time() - t_fin:.3f}s)[/green]")
    
    total_time = time.time() - t_start_total
    n_mapped = sum(len(c) for c in final_parts)
    print(f"\n HydroFlow mapping finished! Mapped {n_mapped} nodes. Total time: {total_time:.3f}s.")
          
    return final_parts, analysis_hist, v_order, i2n