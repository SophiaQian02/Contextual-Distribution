import torch
import torch.nn as nn
import random
import numpy as np
import os
import GPUtil
import time

def compute_5_poly(adj, poly_list):
    poly_arr = []
    for poly in poly_list:
        poly_arr.append(torch.einsum(poly, adj,adj,adj,adj,adj))
    return torch.cat(poly_arr,dim=0)

def compute_6_poly(adj, poly_list):
    poly_arr = []
    n= adj.shape[-1]
    for poly in poly_list:
        if poly[-2:] == 'yz':
            poly_arr.append((torch.einsum(poly[:-2], adj,adj,adj,adj,adj,adj).view(1,1,1).repeat(1,n,n)))
        elif poly[-1] == 'z':
            poly_arr.append(torch.diag_embed(torch.einsum(poly[:-1], adj,adj,adj,adj,adj,adj)))
        else:
            poly_arr.append(torch.einsum(poly, adj,adj,adj,adj,adj,adj))
    return torch.cat(poly_arr,dim=0)

def compute_poly(adj, deg, poly_list):
    if deg == 5:
        poly = compute_5_poly(adj,poly_list)
    elif deg == 6:
        poly = compute_6_poly(adj.to('cuda:1'),poly_list)
    return poly


def compute_vne(adj, k_max):
    if adj.dim() == 3:
        adj = adj.squeeze(0)
    
    N = adj.shape[0]
    vne_list = []
    
    for i in range(N):
        node_vne = []
        nodes = {i}
        visited = {i}
        current_shell = {i}
        
        for k in range(1, k_max + 1):
            next_shell = set()
            for node in current_shell:
                nbrs = (adj[node] > 0).nonzero().view(-1).tolist()
                for nbr in nbrs:
                    if nbr not in visited:
                        visited.add(nbr)
                        next_shell.add(nbr)
            current_shell = next_shell
            
            sub_nodes = list(visited)
            if len(sub_nodes) < 2:
                node_vne.append(0.0)
                continue
            
            sub_adj = adj[sub_nodes][:, sub_nodes]
            deg = sub_adj.sum(dim=1)
            L = torch.diag(deg) - sub_adj
            
            # Move to CPU for stability
            L_cpu = L.cpu()
            try:
                eig = torch.linalg.eigvalsh(L_cpu)
                eig = eig.to(adj.device)
            except:
                eig = torch.zeros(len(sub_nodes)).to(adj.device)
            
            if torch.isnan(eig).any() or torch.isinf(eig).any():
                print(f"NaN/Inf detected in eigenvalues for node {i}, subgraph size {len(sub_nodes)}")
                eig = torch.zeros(len(sub_nodes)).to(adj.device)

            eig = torch.clamp(eig, min=0)
            tr_L = eig.sum()
            if tr_L > 1e-8:
                rho_eig = eig / tr_L
                rho_eig = rho_eig[rho_eig > 1e-12]
                # Renormalize to ensure sum is 1 after filtering
                rho_sum = rho_eig.sum()
                if rho_sum > 0:
                    rho_eig = rho_eig / rho_sum
                    entropy = -(rho_eig * torch.log(rho_eig)).sum()
                    node_vne.append(entropy.item())
                else:
                    node_vne.append(0.0)
            else:
                node_vne.append(0.0)
        
        vne_list.append(node_vne)
        
    return torch.tensor(vne_list)

def compute_rwse(adj, k_max):
    if adj.dim() == 3:
        adj = adj.squeeze(0)
    
    N = adj.shape[0]
    
    # Degree matrix
    deg = adj.sum(dim=1)
    deg_inv = deg.pow(-1)
    deg_inv[deg_inv == float('inf')] = 0
    P = torch.diag(deg_inv) @ adj
    
    rwse_list = []
    P_k = P.clone()
    
    for k in range(k_max):
        rwse_list.append(torch.diag(P_k))
        P_k = P_k @ P
        
    rwse = torch.stack(rwse_list, dim=1) # (N, k_max)
    return rwse

def compute_lappe(adj, k_max):
    if adj.dim() == 3:
        adj = adj.squeeze(0)
    
    N = adj.shape[0]
    
    # Normalized Laplacian
    deg = adj.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    
    L = torch.eye(N, device=adj.device) - D_inv_sqrt @ adj @ D_inv_sqrt
    
    # Eigendecomposition
    # L is symmetric, so use eigh
    try:
        eigvals, eigvecs = torch.linalg.eigh(L)
    except:
        # Fallback if eigendecomposition fails
        return torch.zeros(N, k_max, device=adj.device)
    
    # Sort eigenvalues and eigenvectors
    idx = eigvals.argsort()
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    lappe_list = []
    # Use the first k_max eigenvectors (including the trivial one)
    # Or should we skip the first one? Usually LapPE uses the smallest non-trivial ones.
    # But if we want positional encoding, the first one (constant) is also informative about connected components?
    # Standard LapPE in Graph Transformers often uses the first k smallest eigenvectors.
    # Let's use the first k_max.
    
    for i in range(min(N, k_max)):
        vec = eigvecs[:, i]
        lappe_list.append(torch.abs(vec))
        
    # Pad if N < k_max
    if N < k_max:
        for _ in range(k_max - N):
            lappe_list.append(torch.zeros(N, device=adj.device))
            
    lappe = torch.stack(lappe_list, dim=1) # (N, k_max)
    return lappe


def set_gpu(gpu):
    if gpu[0] == '-':
        deviceIDs = []
        count = 0
        num_gpus = int(gpu[1:])
        print('searching for available gpu...')
        deviceIDs = GPUtil.getAvailable(order='memory', limit=8, maxLoad=0.5, maxMemory=0.5, includeNan=False, excludeID=[],
                                            excludeUUID=[])
        print(deviceIDs)
        while len(deviceIDs) < num_gpus:
            time.sleep(60)
            count += 60
            print('Pending... | {} mins |'.format(count//60))
            deviceIDs = GPUtil.getAvailable(order='memory', limit=8, maxLoad=0.5, maxMemory=0.5, includeNan=False, excludeID=[],
                                            excludeUUID=[])
            print(deviceIDs)
        gpu = deviceIDs[0:num_gpus]
    os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(gpu)[1:-1]#.replace(' ','')

def get_name(cfg):
    name = cfg.dataset + '_poly_ppgn_' + cfg.model + '_'
    if cfg.use_poly_data:
        name = name + '{}_expressive_'.format(cfg.d_expressive)
        if cfg.normalize_features:
            name = name + 'feature_normalized' + '_'
    else:
        name = name + 'baseline_'
    name = name + 'depth_{}_hidden_dim_{}_'.format(cfg.depth,cfg.hidden_dim)
    if cfg.subset and (cfg.dataset == 'ZINC'):
        name = name + 'subset'
    else:
        name = name + 'full'
    return name

def set_seed(cfg):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    return

def update_cfg_dims(cfg):
    if cfg.use_poly_data:
        if cfg.d_expressive == 5:
            cfg.input_dim += 1
        elif cfg.d_expressive == 6:
            if cfg.dataset == 'SR':
                cfg.input_dim += 12
            else:
                cfg.input_dim += 9
    if cfg.use_subgraph_entropy:
        cfg.input_dim += cfg.subgraph_entropy_k
    if hasattr(cfg, 'use_rwse') and cfg.use_rwse:
        cfg.input_dim += cfg.rwse_k
    if hasattr(cfg, 'use_lappe') and cfg.use_lappe:
        cfg.input_dim += cfg.lappe_k
    if hasattr(cfg, 'use_kernel_features') and cfg.use_kernel_features:
        cfg.input_dim += cfg.kernel_m
    return cfg

def compute_kernel_features(adj, k, m):
    if adj.dim() == 3:
        adj = adj.squeeze(0)
    
    # Compute VNE for k-hop subgraphs
    # compute_vne returns (N, k_max)
    vne_all = compute_vne(adj, k)
    # vne_all is on CPU likely, move to adj.device
    vne_all = vne_all.to(adj.device)
    
    # The k-th hop VNE is at index k-1
    H = vne_all[:, k-1]
    
    # Handle potential NaNs/Infs in H
    if torch.isnan(H).any() or torch.isinf(H).any():
        print("NaN/Inf detected in H (VNE values)")
        H = torch.where(torch.isnan(H) | torch.isinf(H), torch.zeros_like(H), H)

    N = adj.shape[0]
    kernel_features = torch.zeros(N, m).to(adj.device)
    
    for u in range(N):
        nbrs = (adj[u] > 0).nonzero().view(-1).tolist()
        
        if len(nbrs) == 0:
            continue
            
        k_values = []
        for v in nbrs:
            # Formula: 1/2 * exp(-|H(u) - H(v)|)
            val = 0.5 * torch.exp(-torch.abs(H[u] - H[v]))
            k_values.append(val)
        
        k_values = torch.stack(k_values)
        
        if len(k_values) >= m:
            # Sample m neighbors
            perm = torch.randperm(len(k_values))
            idx = perm[:m]
            selected = k_values[idx]
        else:
            # Pad with 0
            padding = torch.zeros(m - len(k_values)).to(adj.device)
            selected = torch.cat([k_values, padding])
            
        kernel_features[u] = selected
        
    return kernel_features
