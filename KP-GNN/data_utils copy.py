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


def extract_multi_hop_neighbors(data, K, max_edge_attr_num, kernel, t=5, kernel2='reproduce', sample=True):
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
    else:
        # skip 0, 1 as it is the mask and self-loop defined in the model
        edge_attr = (torch.ones([edge_index.size(-1)]) * 2).long()  # E

    adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)
    adj_numpy = torch.from_numpy(adj.toarray()).long()
    edge_attr_adj = torch.from_numpy(to_scipy_sparse_matrix(edge_index, edge_attr, num_nodes).toarray()).long()
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

    kernel_emb, node_entropy = get_k_vec(exist_adj_list, adj_numpy, adj_list, num_nodes, data.node_tags, K, t, data.deg, kernel2, sample)
    #print(kernel_emb)
    data.edge_index = edge_index
    data.edge_attr = edge_attr
    data.kernel_emb = kernel_emb
    data.rep_nodes = t
    data.node_entropy = node_entropy
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


def get_path_entropy(sub_adj):
    sub_adj_np = sub_adj.numpy()
    G_sub = nx.from_numpy_array(sub_adj_np)

    # Compute all-pairs shortest paths within the subgraph
    path_length_dict = dict(nx.all_pairs_shortest_path_length(G_sub))

    # Flatten and count path length frequencies
    lengths = []
    for source, targets in path_length_dict.items():
        for target, dist in targets.items():
            if source != target:  # exclude self-loops
                lengths.append(dist)

    if len(lengths) == 0:
        path_entropy = 0.0
    else:
        length_tensor = torch.tensor(lengths)
        unique_lens, counts = torch.unique(length_tensor, return_counts=True)
        probs = counts.float() / counts.sum()
        path_entropy = -(probs * torch.log(probs + 1e-10)).sum()
        
    return path_entropy

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
    return np.exp(-np.linalg.norm(ents_i - ents_j, ord=1))/2

def get_edge_entropy(sub_adj):
    """
    Compute edge entropy for a subgraph using the provided formula.
    sub_adj: torch.Tensor, adjacency matrix of the subgraph
    """
    # Ensure sub_adj is binary and undirected
    sub_adj = (sub_adj > 0).float()
    V = sub_adj.size(0)
    # Only count upper triangle to avoid double-counting undirected edges
    edge_indices = torch.triu_indices(V, V, offset=1)
    u_list, v_list = edge_indices
    edge_mask = sub_adj[u_list, v_list] > 0
    u_list = u_list[edge_mask]
    v_list = v_list[edge_mask]
    E = u_list.size(0)
    if E == 0:
        return 0.0

    deg = sub_adj.sum(dim=1)
    du = deg[u_list]
    dv = deg[v_list]

    term1 = 1.0 / E
    term2 = 1.0 / (V * E)
    term3 = 1.0 / (E * V * V) * (1.0 / (du * dv))
    edge_entropy = term1 - term2 - term3
    # Sum over all edges
    total_entropy = edge_entropy.sum().item()
    return total_entropy

def get_edge_entropy_MB(sub_adj, beta=1.0):
    """
    Compute edge entropy for a subgraph using Maxwell-Boltzmann statistics.
    sub_adj: torch.Tensor, adjacency matrix of the subgraph
    beta: float, inverse temperature parameter
    Returns:
        edge_entropy: torch.Tensor, shape (V, V), edge entropy matrix
    """
    sub_adj = sub_adj.float()  # Ensure binary adjacency matrix
    V = sub_adj.size(0)
    N = V
    # Compute Laplacian
    deg = sub_adj.sum(dim=1)
    L = torch.diag(deg) - sub_adj

    # Eigendecomposition
    lambdas, eigvecs = torch.linalg.eigh(L)  # lambdas: (V,), eigvecs: (V, V)

    # Maxwell-Boltzmann weights
    exp_vals = torch.exp(-beta * lambdas)
    Z = exp_vals.sum()
    probs = exp_vals / Z
    sigma_MB = -N * probs * torch.log(probs + 1e-12)  # shape: (V,)

    # Edge entropy projection
    # For each i: sigma_MB[i] * (eigvecs[:, i] @ eigvecs[:, i].T)
    # So sum_i sigma_MB[i] * outer(eigvecs[:, i], eigvecs[:, i])
    edge_entropy = 0
    for i in range(V):
        phi = eigvecs[:, i].unsqueeze(1)  # (V, 1)
        edge_entropy += sigma_MB[i] * (phi.T @ phi)  # (V, V)

    return edge_entropy
    

def get_k_vec(exist_adj_list, adj_ori, adj_list, num_nodes, node_tags, K, t, degree_as_tag, kernel2, sample=True):
    
    node_t_vec = torch.zeros((num_nodes, K, t), dtype=torch.float32)
    node_entropy_K = torch.zeros((num_nodes, K), dtype=torch.float32)
    node_kernel_lists =[]
    for i in range(K):
        node_entropy = torch.zeros(num_nodes, dtype= torch.float32)
        adj_entropy = torch.zeros(num_nodes, dtype= torch.float32)
        node_numbers = torch.zeros(num_nodes, dtype= torch.float32)

        adj_i = exist_adj_list[i]
        
        for node in range(num_nodes):
            neighbors = torch.where(adj_i[node] > 0)[0]
            num_sub_nodes = neighbors.size(-1)
            if num_sub_nodes <=2:
                node_entropy[node] = 0.0
                continue
            
            sub_adj = adj_ori[neighbors][:, neighbors]
            adj_entropy[node] = get_adj_entropy(sub_adj)
            #path_entropy[node] = get_path_entropy(sub_adj)
            # edge_entropy = get_edge_entropy(sub_adj)
            # #edge_entropy_MB = get_edge_entropy_MB(sub_adj)
            # sub_degrees = sub_adj.sum(dim=0)
            
            # if degree_as_tag:
            #     p = sub_degrees / (sub_degrees.sum() + 1e-10)
            # else:
            #     node_tags = torch.tensor(node_tags)
            #     sub_tags = node_tags[neighbors]  
            #     degree_tag_pairs = torch.stack([sub_degrees, sub_tags], dim=1)
            #     unique_pairs, counts = torch.unique(degree_tag_pairs, dim=0, return_counts=True)
            #     joint_p = counts.float() / counts.sum()
            #     p = joint_p
            # node_entropy[node] = -(p * torch.log(p + 1e-10)).sum()
            # node_numbers[node] = len(neighbors) 
            
        node_entropy_K[:, i] = adj_entropy
        
        adj_order_i = adj_list[i]
        adj_order_i.fill_diagonal_(1)
        
        if sample == False:
            
            node_kernel_list = []
            # print(1)
            pass
            # node_kernel_list = []
            # for node in range(num_nodes):
                
            #     neighbors = torch.where(adj_order_i[node] > 0)[0]
            #     num_sub_nodes = neighbors.size(-1)
            #     if num_sub_nodes <2:
            #         node_kernel_list.append(torch.zeros(1))
            #         continue
                
            #     neighbor_entropy = node_entropy[neighbors]
            #     if kernel2 == 'js':
            #         center_ent = node_entropy[node]
            #         center_num = node_numbers[node]
            #         neighbor_ents = node_entropy[neighbors]
            #         neighbor_nums = node_numbers[neighbors]
            #         ents_i = center_ent.expand_as(neighbor_ents)
            #         ents_j = neighbor_ents
            #         num_i = center_num.expand_as(neighbor_nums)
            #         num_j = neighbor_nums
            #         kernel_vals = js_kernel(num_i, num_j, ents_i, ents_j)
            #         # mask = ~torch.eye(num_sub_nodes, dtype=torch.bool)
            #         # kernel_vals = js_kernel[mask]
            #         #print(type(kernel_vals))
            #         node_kernel_list.append(kernel_vals)
            #     elif kernel2 == 'reproduce':
            #         center_ent = node_entropy[node]
            #         neighbor_ents = node_entropy[neighbors]
            #         diff = torch.abs(neighbor_ents - center_ent)
            #         kernel_mat = 0.5 * torch.exp(-diff)
            #         # mask = ~torch.eye(len(neighbors), dtype=torch.bool)
            #         # kernel_vals = kernel_mat[mask]
            #         node_kernel_list.append(kernel_mat)
            # node_kernel_lists.append(node_kernel_list)

        else:

            # sorted_entropy, indices = torch.sort(node_entropy, descending=True)
            
            # top_indices = indices[:t]

            # top_indices = epison_sampling(node_entropy, t)
            # nums = node_numbers
            # ents = node_entropy
            # mask = (nums > 1).unsqueeze(1) & (nums > 1).unsqueeze(0) #broadcast
            # nums_i = nums.unsqueeze(1).expand(-1, num_nodes)  
            # nums_j = nums.unsqueeze(0).expand(num_nodes, -1)  
            # ents_i = ents.unsqueeze(1).expand(-1, num_nodes)
            # ents_j = ents.unsqueeze(0).expand(num_nodes, -1)
            
            # if kernel2 == 'js':
            #     js_value = js_kernel(nums_i, nums_j, ents_i, ents_j)
            # elif kernel2 == 'reproduce':
            #     js_value = reproduce_kernel(ents_i, ents_j)

            # adj_js = js_value * mask.float()    
            # diag = torch.diag(adj_js)  
            # denominator = torch.sqrt(diag.unsqueeze(1) * diag.unsqueeze(0)) + 1e-10   
            # adj_js = adj_js / denominator
            # node_t_vec[:, i, :] = adj_js[:, top_indices]
            
            for node in range(num_nodes):
                neighbors = torch.where(adj_order_i[node]>0)[0]
            
                if kernel2 == 'js':
                        center_ent = adj_entropy[node]
                        center_num = node_numbers[node]
                        neighbor_ents = adj_entropy[neighbors]
                        neighbor_nums = node_numbers[neighbors]
                        ents_i = center_ent.expand_as(neighbor_ents)
                        ents_j = neighbor_ents
                        num_i = center_num.expand_as(neighbor_nums)
                        num_j = neighbor_nums
                        kernel_vals = js_kernel(num_i, num_j, ents_i, ents_j)
                elif kernel2 == 'reproduce':
                        center_ent = adj_entropy[node]
                        neighbor_ents = adj_entropy[neighbors]
                        diff = torch.abs(neighbor_ents - center_ent)
                        kernel_vals = 0.5 * torch.exp(-diff)
                        
                num_sub_nodes = neighbors.size(0)
                
                if num_sub_nodes >= t:
                    probs = kernel_vals / kernel_vals.sum()
                    sampled_indices = torch.multinomial(probs, t, replacement=False)
                    sampled_vals = kernel_vals[sampled_indices]
                else:
                    sampled_vals = torch.zeros(t)
                    sampled_vals[:num_sub_nodes] = kernel_vals
    
                node_t_vec[node, i, :] = sampled_vals
                    
            
    if sample:
        
        return node_t_vec, node_entropy_K
    else:
        #print(node_entropy_K)
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
