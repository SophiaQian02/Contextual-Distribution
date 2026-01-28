"""
utils for processing data used for training and evaluation
"""
import itertools
from copy import deepcopy as c

import networkx as nx
import numpy as np
import scipy.sparse as ssp
import torch
from scipy import linalg
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_scipy_sparse_matrix


def maybe_num_nodes(index, num_nodes=None):
    return index.max().item() + 1 if num_nodes is None else num_nodes


def extract_multi_hop_neighbors(data, K, max_edge_attr_num, kernel, t=5, kernel2='reproduce', sample=True, se_type='entropy', use_edge_kernel=False):
    """generate multi-hop neighbors for input PyG graph using shortest path distance kernel
    Args:
        data (torch_geometric.data.Data): PyG graph data instance
        K (int): number of hop

        kernel (str): kernel used to extract neighbors
    """
    assert (isinstance(data, Data))
    x, edge_index, num_nodes = data.x, data.edge_index, data.num_nodes

    # graph with no edge
    # if edge_index.size(1) == 0:
    #     edge_matrix_size = [num_nodes, K, max_edge_type, 2]
    #     configuration_matrix_size = [num_nodes, K, max_hop_num]
    #     peripheral_edge_matrix = torch.zeros(edge_matrix_size, dtype=torch.long)
    #     peripheral_configuration_matrix = torch.zeros(configuration_matrix_size, dtype=torch.long)
    #     data.peripheral_edge_attr = peripheral_edge_matrix
    #     data.peripheral_configuration = peripheral_configuration_matrix
    #     return data

    if "edge_attr" in data:
        edge_attr = data.edge_attr
        if len(edge_attr.shape) > 1:
             edge_attr_for_adj = edge_attr[:, 0]
        else:
             edge_attr_for_adj = edge_attr
    else:
        # skip 0, 1 as it is the mask and self-loop defined in the model
        edge_attr = (torch.ones([edge_index.size(-1)]) * 2).long()  # E
        edge_attr_for_adj = edge_attr

    adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)
    adj_numpy = torch.from_numpy(adj.toarray()).long()
    edge_attr_adj = torch.from_numpy(to_scipy_sparse_matrix(edge_index, edge_attr_for_adj, num_nodes).toarray()).long()
    # compute each order of adj
    adj_list = adj_K_order(adj, K)

    exist_adj_list = []
    if kernel == "gd":
        # create K-hop edge with graph diffusion kernel
        final_adj = 0
        for adj_ in adj_list:
            final_adj += adj_
            final_adj[final_adj>1] = 1
            
            adj_c = c(final_adj)
            adj_c.fill_diagonal_(1)
            exist_adj_list.append(adj_c)
    else:
        # process adj list to generate shortest path distance matrix with path number
        exist_adj = c(adj_list[0])
        adj_c = c(exist_adj)
        adj_c.fill_diagonal_(1)
        exist_adj_list.append(adj_c)
        for i in range(1, len(adj_list)):
            adj_ = c(adj_list[i])
            # mask all the edge that already exist in previous hops
            adj_[exist_adj > 0] = 0
            exist_adj = exist_adj + adj_
            exist_adj[exist_adj > 1] = 1
            adj_c = c(exist_adj)
            adj_c.fill_diagonal_(1)
            exist_adj_list.append(adj_c)
            adj_list[i] = adj_
        # create K-hop edge with sortest path distance kernel
        final_adj = exist_adj

    g = nx.from_numpy_array(final_adj.numpy(), create_using=nx.DiGraph)
    edge_list = g.edges
    edge_index = torch.from_numpy(np.array(edge_list).T).long()

    hop1_edge_attr = edge_attr_adj[edge_index[0, :], edge_index[1, :]]
    edge_attr_list = [hop1_edge_attr.unsqueeze(-1)]
    pe_attr_list = []
    for i in range(1, len(adj_list)):
        adj_ = c(adj_list[i])
        adj_[adj_ > max_edge_attr_num] = max_edge_attr_num
        # skip 1 as it is the self-loop defined in the model
        adj_[adj_ > 0] = adj_[adj_ > 0] + 1
        adj_ = adj_.long()
        hopk_edge_attr = adj_[edge_index[0, :], edge_index[1, :]].unsqueeze(-1)
        edge_attr_list.append(hopk_edge_attr)
        pe_attr_list.append(torch.diag(adj_).unsqueeze(-1))
    edge_attr = torch.cat(edge_attr_list, dim=-1)  # E * K
    if K > 1:
        pe_attr = torch.cat(pe_attr_list, dim=-1)  # N * K-1
    else:
        pe_attr = None

    
    #t = min(t, data.min_node_number)
    
    #print(data.min_node_number)

    kernel_emb, node_entropy = get_k_vec(exist_adj_list, adj_numpy, adj_list, num_nodes, K, t, data.deg, kernel2, sample, se_type)
    #print(kernel_emb)
    
    if use_edge_kernel:
        entropy_kernel_values = []
        for edge in edge_index.T:
            node_i, node_j = edge[0].item(), edge[1].item()
            
            # For each hop, compute the kernel value between the entropy of node_i and node_j
            kernel_values_for_edge = []
            for hop in range(K):
                entropy_i = node_entropy[node_i, hop]  # Entropy for node_i at hop 'hop'
                entropy_j = node_entropy[node_j, hop]  # Entropy for node_j at hop 'hop'
                
                
                kernel_value = reproduce_kernel(entropy_i, entropy_j)
                kernel_values_for_edge.append(kernel_value)
            
            # Append the kernel values for this edge to the list
            entropy_kernel_values.append(kernel_values_for_edge)
        
        # Convert the entropy kernel values to a tensor and add them as additional edge attributes
        entropy_kernel_tensor = torch.tensor(entropy_kernel_values).float()  # Shape: [E, K]
        #print(entropy_kernel_tensor.shape)
        #edge_attr = torch.cat([edge_attr, entropy_kernel_tensor], dim=-1) 
        #edge_attr = edge_attr + entropy_kernel_tensor
    else:
        entropy_kernel_tensor = torch.zeros(edge_index.size(1), K, t).float()
    
    
    
    
    data.edge_index = edge_index
    data.edge_attr = edge_attr
    data.edge_ker_emb = entropy_kernel_tensor
    data.kernel_emb = kernel_emb
    data.rep_nodes = t
    data.node_entropy = node_entropy
    #print(data.kernel_emb)
    #print(data.node_entropy)
    data.pe_attr = pe_attr
    return data


def js_kernel(V_p, V_q, H_p, H_q, lambda_param = 1.0):
    
    denominator = 2 * (V_p + V_q) + 1e-10
    weight_p = (2 * V_p - V_q) / denominator
    weight_q = (2 * V_q - V_p) / denominator
    weight_entropy = weight_p * H_p + weight_q * H_q
    return torch.exp(-lambda_param * weight_entropy)

def epison_sampling(node_entropy, t, max_iter = 20, initial_eps=None, verbose=False):
    
    sorted_entropy, indices = torch.sort(node_entropy, descending=True)
    
    if initial_eps is None:
        epsilon = 0.5 * torch.std(node_entropy).item()
    else:
        epsilon = initial_eps
        
    # step = eps * 0.5 # adjust step
    # for it in range(max_iter):
    selected= []
    for idx in indices:
        ent = node_entropy[idx]
        if not selected or all(torch.abs(ent-node_entropy[i]) >= epsilon for i in selected):
            selected.append(idx.item())
        if len(selected) >= t:
            break
                
        # if len(selected) < t:
        #     eps *= 0.8
            
            
    for idx in indices:
        if len(selected) >= t:
            break
        if idx.item() not in selected:
            selected.append(idx.item())
            
    if verbose:
        print(f"[ε-cover] Final ε = {epsilon:.4f}, selected = {len(selected)}")
        
    return torch.tensor(selected, dtype=torch.long)


# def get_path_entropy(sub_adj):
#     sub_adj_np = sub_adj.numpy()
#     G_sub = nx.from_numpy_array(sub_adj_np)

#     # Compute all-pairs shortest paths within the subgraph
#     path_length_dict = dict(nx.all_pairs_shortest_path_length(G_sub))

#     # Flatten and count path length frequencies
#     lengths = []
#     for source, targets in path_length_dict.items():
#         for target, dist in targets.items():
#             if source != target:  # exclude self-loops
#                 lengths.append(dist)

#     if len(lengths) == 0:
#         path_entropy = 0.0
#     else:
#         length_tensor = torch.tensor(lengths)
#         unique_lens, counts = torch.unique(length_tensor, return_counts=True)
#         probs = counts.float() / counts.sum()
#         path_entropy = -(probs * torch.log(probs + 1e-10)).sum()
        
#     return path_entropy

def get_adj_entropy(adj):
    """
    Compute Laplacian spectral entropy from adjacency matrix.
    """
    if isinstance(adj, torch.Tensor):
        adj = adj.cpu().numpy()

    deg = np.diag(adj.sum(axis=1))
    laplacian = deg - adj

    eigenvalues = np.linalg.eigvalsh(laplacian)
    eigenvalues = np.clip(eigenvalues, a_min=1e-10, a_max=None)
    prob = eigenvalues / np.sum(eigenvalues)
    entropy = -np.sum(prob * np.log2(prob))
    return entropy

# def get_adj_entropy(adj):
    
#     eigenvalues = np.linalg.eigvalsh(adj)
    
#     #abs_eigen = eigenvalues
#     abs_eigen = eigenvalues - np.min(eigenvalues)
    
#     prob = abs_eigen / np.sum(abs_eigen)

#     entropy = -np.sum([p*np.log2(p) for p in prob if p >0])
#     #print(entropy)
    
#     return entropy

def reproduce_kernel(ents_i, ents_j):
    #return np.exp(-np.linalg.norm(ents_i - ents_j, ord=1))/2
    return np.exp(-np.abs(ents_i - ents_j))/2

# def get_edge_entropy(sub_adj):
#     """
#     Compute edge entropy for a subgraph using the provided formula.
#     sub_adj: torch.Tensor, adjacency matrix of the subgraph
#     """
#     # Ensure sub_adj is binary and undirected
#     sub_adj = (sub_adj > 0).float()
#     V = sub_adj.size(0)
#     # Only count upper triangle to avoid double-counting undirected edges
#     edge_indices = torch.triu_indices(V, V, offset=1)
#     u_list, v_list = edge_indices
#     edge_mask = sub_adj[u_list, v_list] > 0
#     u_list = u_list[edge_mask]
#     v_list = v_list[edge_mask]
#     E = u_list.size(0)
#     if E == 0:
#         return 0.0

#     deg = sub_adj.sum(dim=1)
#     du = deg[u_list]
#     dv = deg[v_list]

#     term1 = 1.0 / E
#     term2 = 1.0 / (V * E)
#     term3 = 1.0 / (E * V * V) * (1.0 / (du * dv))
#     edge_entropy = term1 - term2 - term3
#     # Sum over all edges
#     total_entropy = edge_entropy.sum().item()
#     return total_entropy

# def get_edge_entropy_MB(sub_adj, beta=1.0):
#     """
#     Compute edge entropy for a subgraph using Maxwell-Boltzmann statistics.
#     sub_adj: torch.Tensor, adjacency matrix of the subgraph
#     beta: float, inverse temperature parameter
#     Returns:
#         edge_entropy: torch.Tensor, shape (V, V), edge entropy matrix
#     """
#     sub_adj = sub_adj.float()  # Ensure binary adjacency matrix
#     V = sub_adj.size(0)
#     N = V
#     # Compute Laplacian
#     deg = sub_adj.sum(dim=1)
#     L = torch.diag(deg) - sub_adj

#     # Eigendecomposition
#     lambdas, eigvecs = torch.linalg.eigh(L)  # lambdas: (V,), eigvecs: (V, V)

#     # Maxwell-Boltzmann weights
#     exp_vals = torch.exp(-beta * lambdas)
#     Z = exp_vals.sum()
#     probs = exp_vals / Z
#     sigma_MB = -N * probs * torch.log(probs + 1e-12)  # shape: (V,)

#     # Edge entropy projection
#     # For each i: sigma_MB[i] * (eigvecs[:, i] @ eigvecs[:, i].T)
#     # So sum_i sigma_MB[i] * outer(eigvecs[:, i], eigvecs[:, i])
#     edge_entropy = 0
#     for i in range(V):
#         phi = eigvecs[:, i].unsqueeze(1)  # (V, 1)
#         edge_entropy += sigma_MB[i] * (phi.T @ phi)  # (V, V)

#     return edge_entropy
    

def get_rwse(adj, K, num_nodes):
    """
    Compute Random Walk Structural Encoding (RWSE).
    Returns the diagonal of the random walk matrix powers.
    """
    adj = adj.float()
    deg = adj.sum(dim=1)
    deg_inv = deg.pow(-1)
    deg_inv[deg_inv == float('inf')] = 0
    P = torch.diag(deg_inv) @ adj
    
    rwse = torch.zeros((num_nodes, K), dtype=torch.float32)
    P_k = P
    for k in range(K):
        rwse[:, k] = torch.diag(P_k)
        if k < K - 1:
            P_k = P_k @ P
            
    return rwse

import torch

def get_lapse(adj, K, num_nodes):
    """
    Compute Laplacian Eigenvector Positional Encodings (LapPE) as defined in the paper.
    Returns the absolute value of the l2-normalized eigenvectors associated with non-trivial eigenvalues.
    
    Args:
        adj: Adjacency matrix (n x n)
        K: Number of eigenvectors to return
        num_nodes: Number of nodes (n)
        
    Returns:
        lapse: Tensor of shape (num_nodes, K) containing the LapPEs.
    """
    # Ensure adj is float
    adj = adj.float()
    
    # Step 1: Compute Degree Matrix D
    deg = adj.sum(dim=1)  # Shape: (n,)
    D = torch.diag(deg)   # Shape: (n, n)
    
    # Step 2: Compute Unnormalized Laplacian L = D - A
    L = D - adj  # Shape: (n, n)
    
    # Step 3: Eigen decomposition
    # torch.linalg.eigh returns eigenvalues in ascending order
    vals, vecs = torch.linalg.eigh(L)
    
    # Step 4: Find non-trivial eigenvalues (lambda_i != 0)
    # We use a small epsilon for numerical stability
    eps = 1e-6
    non_trivial_mask = vals > eps
    
    # Get the corresponding eigenvectors
    non_trivial_vecs = vecs[:, non_trivial_mask]  # Shape: (n, m), where m is number of non-trivial eigenvectors
    
    # Step 5: Take the first K non-trivial eigenvectors (or as many as available)
    k_eff = min(K, non_trivial_vecs.shape[1])
    selected_vecs = non_trivial_vecs[:, :k_eff]  # Shape: (n, k_eff)
    
    # Step 6: L2 normalize each eigenvector
    # Normalize along dimension 0 (over nodes)
    norm = torch.norm(selected_vecs, p=2, dim=0, keepdim=True)  # Shape: (1, k_eff)
    normalized_vecs = selected_vecs / norm  # Shape: (n, k_eff)
    
    # Step 7: Take absolute value
    lappe = torch.abs(normalized_vecs)  # Shape: (n, k_eff)
    
    # Step 8: Pad with zeros if needed (if K > number of non-trivial eigenvectors)
    if k_eff < K:
        padding = torch.zeros((num_nodes, K - k_eff), dtype=torch.float32)
        lappe = torch.cat([lappe, padding], dim=1)  # Shape: (n, K)
    
    return lappe

def get_k_vec(exist_adj_list, adj_ori, adj_list, num_nodes, K, t, degree_as_tag, kernel2, sample=True, se_type='entropy'):
    
    node_t_vec = torch.zeros((num_nodes, K, t), dtype=torch.float32)
    node_entropy_K = torch.zeros((num_nodes, K), dtype=torch.float32)
    
    if se_type == 'none':
        return node_t_vec, node_entropy_K
    
    # 1. Compute Node Entropy / Structural Encoding for all K hops
    if se_type == 'RWSE':
        node_entropy_K = get_rwse(adj_ori, K, num_nodes)
    elif se_type == 'LapSE':
        node_entropy_K = get_lapse(adj_ori, K, num_nodes)
    else: # se_type == 'entropy'
        for i in range(K):
            adj_i = exist_adj_list[i]
            
            node_entropy = torch.zeros(num_nodes, dtype=torch.float32)
            adj_entropy = torch.zeros(num_nodes, dtype=torch.float32)
            
            # Compute entropy for hop i
            for node in range(num_nodes):
                neighbors = torch.where(adj_i[node] > 0)[0]
                num_sub_nodes = neighbors.size(-1)
                
                if num_sub_nodes == 0:
                    node_entropy[node] = 0.0
                    continue
                
                sub_adj = adj_ori[neighbors][:, neighbors]
                adj_entropy[node] = get_adj_entropy(sub_adj)
                
            node_entropy_K[:, i] = adj_entropy

    # 2. Kernel Construction & 3. Feature Extraction (Image Logic)
    # Calculate averaged kernel matrix and extract top-t neighbors
    
    for node in range(num_nodes):
        # Direct neighbors (1-hop)
        # adj_ori is dense adjacency, use it to find neighbors
        neighbors = torch.where(adj_ori[node] > 0)[0]
        
        if len(neighbors) == 0:
            continue
            
        # Get entropies for center and neighbors across all K hops
        # center_ents: [K]
        # neighbor_ents: [num_neighbors, K]
        center_ents = node_entropy_K[node]
        neighbor_ents = node_entropy_K[neighbors]
        
        # Calculate diffs: |H_i^k - H_j^k|
        # Shape: [num_neighbors, K]
        diffs = torch.abs(center_ents.unsqueeze(0) - neighbor_ents)
        
        # Calculate Kernel values: 0.5 * exp(-diff)
        # Note: Using standard formula from image. 
        k_vals = 0.5 * torch.exp(-diffs)
        
        # Calculate Averaged Kernel Value over K hops
        # Shape: [num_neighbors]
        k_val_avg = k_vals.mean(dim=1)
        
        # Extract Top-t values
        num_sub_nodes = len(neighbors)
        if num_sub_nodes >= t:
            vals, _ = torch.sort(k_val_avg, descending=True)
            sampled_vals = vals[:t]
        else:
            sampled_vals = torch.zeros(t)
            vals, _ = torch.sort(k_val_avg, descending=True)
            sampled_vals[:num_sub_nodes] = vals
            
        # Assign to node_t_vec
        # Since this feature vector is derived from averaging K hops, it's a single vector [t].
        # We replicate it across the K dimension to match output shape [N, K, t]
        for k in range(K):
             node_t_vec[node, k, :] = sampled_vals

    return node_t_vec, node_entropy_K

def adj_K_order(adj, K):
    """compute the K order of adjacency given scipy matrix
    adj (coo_matrix): adjacency matrix
    K (int): number of hop
    """
    adj_list = [c(adj)]
    for i in range(K - 1):
        adj_ = adj_list[-1] @ adj
        adj_list.append(adj_) 

    for i, adj_ in enumerate(adj_list):
        adj_ = torch.from_numpy(adj_.toarray()).int()
        # prevent the precision overflow
        # adj_[adj_<0]=1e8
        adj_.fill_diagonal_(0)  # not rooted subgraph
        adj_list[i] = adj_
    return adj_list



def nx_compute_shortest_path_length(G, max_length):
    """Compute all pair shortest path length in the graph
    Args:
        G (networkx): input graph
        max_length (int): max length when computing shortest path

    """
    num_node = G.number_of_nodes()
    shortest_path_length_matrix = torch.zeros([num_node, num_node]).int()
    all_shortest_path_lengths = nx.all_pairs_shortest_path_length(G, max_length)
    for shortest_path_lengths in all_shortest_path_lengths:
        index, path_lengths = shortest_path_lengths
        for end_node, path_length in path_lengths.items():
            if end_node == index:
                continue
            else:
                shortest_path_length_matrix[index, end_node] = path_length
    return shortest_path_length_matrix


def to_dense_edge_feature(edge_feature, edge_index, num_nodes):
    """Convert edge feature to dense adj
    Args:
        edge_feature (torch.tensor): original edge feature
        edge_index (torch.tensor): edge index
        num_nodes (int): number of node in graph
    """
    edge_feature = edge_feature.squeeze()
    K = list(edge_feature.size()[1:])
    adj = torch.zeros(list(itertools.chain.from_iterable([[num_nodes], [num_nodes], K])), dtype=edge_feature.dtype)
    for i in range(edge_index.size(-1)):
        v = edge_index[0, i]
        u = edge_index[1, i]
        adj[v, u] = edge_feature[i]

    return adj


def PyG_collate(examples):
    """PyG collcate function
    Args:
        examples(list): batch of samples
    """
    data = Batch.from_data_list(examples)
    return data


def PyG_collate_new(examples):
    data = Batch.from_data_list(examples)
    num_data_nodes = data.num_data_nodes
    node_to_batch = list(itertools.chain.from_iterable(
        [[i for _ in range(num_node.item())] for i, num_node in enumerate(num_data_nodes)]))
    data.batch = torch.tensor(node_to_batch)
    return data


def resistance_distance(data):
    """resistance distance.See "Link prediction in complex networks: A survey".
    Adapted from NestedGNN:https://github.com/muhanzhang/NestedGNN
    Args:
        data(PyG.Data):pyg data object
    """

    edge_index = data.edge_index
    num_nodes = data.num_nodes
    adj = to_scipy_sparse_matrix(
        edge_index, num_nodes=num_nodes
    ).tocsr()
    laplacian = ssp.csgraph.laplacian(adj).toarray()
    try:
        L_inv = linalg.pinv(laplacian)
    except:
        laplacian += 0.01 * np.eye(*laplacian.shape)
    lxx = L_inv[0, 0]
    lyy = L_inv[list(range(len(L_inv))), list(range(len(L_inv)))]
    lxy = L_inv[0, :]
    lyx = L_inv[:, 0]
    rd_to_x = torch.FloatTensor((lxx + lyy - lxy - lyx)).unsqueeze(1)
    data.rd = rd_to_x
    return data


def post_transform(wo_path_encoding, wo_edge_feature):
    """Post transformation of dataset for KP-GNN
    Args:
        wo_path_encoding (bool): If true, remove path encoding from model
        wo_edge_feature (bool): If true, remove edge feature from model
    """
    if wo_path_encoding and wo_edge_feature:
        def transform(g):
            edge_attr = g.edge_attr
            edge_attr[edge_attr > 2] = 2
            g.edge_attr = edge_attr
            if "pe_attr" in g:
                pe_attr = g.pe_attr
                pe_attr[pe_attr > 0] = 0
                g.pe_attr = pe_attr
            return g
    elif wo_edge_feature:
        def transform(g):
            edge_attr = g.edge_attr
            t = edge_attr[:, 0]
            t[t > 2] = 2
            edge_attr[:, 0] = t
            g.edge_attr = edge_attr
            return g

    elif wo_path_encoding:
        def transform(g):
            edge_attr = g.edge_attr
            t = edge_attr[:, 1:]
            t[t > 2] = 2
            edge_attr[:, 1:] = t
            g.edge_attr = edge_attr
            if "pe_attr" in g:
                pe_attr = g.pe_attr
                pe_attr[pe_attr > 0] = 0
                g.pe_attr = pe_attr
            return g
    else:
        def transform(g):
            return g

    return transform
