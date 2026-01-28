import torch
import networkx as nx
import numpy as np

def get_adj_entropy(adj):
    if isinstance(adj, torch.Tensor):
        adj = adj.cpu().numpy()

    deg = np.diag(adj.sum(axis=1))
    laplacian = deg - adj

    eigenvalues = np.linalg.eigvalsh(laplacian)
    eigenvalues = np.clip(eigenvalues, a_min=1e-10, a_max=None)
    prob = eigenvalues / np.sum(eigenvalues)
    entropy = -np.sum(prob * np.log2(prob))
    return float(entropy)


def node_khop_entropy_from_nx(G: nx.Graph, K: int):

    N = G.number_of_nodes()
    A = nx.to_numpy_array(G)              
    A_t = torch.from_numpy(A).float()

    entropies = np.zeros((N, K), dtype=np.float32)

    
    for node in range(N):
       
        lengths = nx.single_source_shortest_path_length(G, node, cutoff=K)
        for k in range(1, K + 1):
            neigh = [v for v, dist in lengths.items() if dist <= k]
            if len(neigh) <= 2:
                continue

            sub_adj = A_t[neigh][:, neigh]      
            entropies[node, k - 1] = get_adj_entropy(sub_adj)

    return entropies   

def compute_rwse(adj, K):
    """
    Compute Random Walk Structural Encoding (RWSE) for a graph.
    RWSE_k = diag(P^k), where P = D^-1 A.
    Returns a matrix of shape (N, K) where the k-th column is the diagonal of P^(k+1).
    """
    # adj is adjacency matrix (N, N)
    deg = np.sum(adj, axis=1)
    deg_inv = np.zeros_like(deg)
    deg_inv[deg != 0] = 1.0 / deg[deg != 0]
    D_inv = np.diag(deg_inv)
    P = np.matmul(D_inv, adj)
    
    rwse_features = []
    P_k = P
    for k in range(K):
        # k=0 -> P^1, k=1 -> P^2 ...
        rwse_features.append(np.diag(P_k))
        P_k = np.matmul(P_k, P)
        
    return np.stack(rwse_features, axis=1) # (N, K)

def compute_lapse(adj, K):
    """
    Compute Laplacian Positional Encoding (LapSE) for a graph.
    LapPE_i = |normalize(U_i)| where U_i is the i-th eigenvector of the Normalized Laplacian.
    We take the first K non-trivial eigenvectors (skipping the first one if connected, but usually just smallest K).
    Here we take the smallest K eigenvectors corresponding to the smallest eigenvalues.
    """
    # Normalized Laplacian: L = I - D^-1/2 A D^-1/2
    deg = np.sum(adj, axis=1)
    deg_inv_sqrt = np.zeros_like(deg)
    deg_inv_sqrt[deg != 0] = 1.0 / np.sqrt(deg[deg != 0])
    D_inv_sqrt = np.diag(deg_inv_sqrt)
    
    I = np.eye(adj.shape[0])
    L = I - np.matmul(np.matmul(D_inv_sqrt, adj), D_inv_sqrt)
    
    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    
    # Sort just in case (eigh usually returns sorted)
    idx = eigenvalues.argsort()
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Take first K eigenvectors. 
    # Note: The first eigenvalue is 0. The corresponding eigenvector is constant (scaled by degree).
    # Often LapPE skips the first one (k=1 to K+1). 
    # However, without specific instruction, taking the smallest K is standard for "first K eigenvectors".
    # Let's take indices 1 to K+1 (skipping the 0-th trivial one) to capture structure, 
    # as the 0-th one carries little info for position differentiation in connected components.
    # But if K is large, we might run out of eigenvectors.
    
    N = adj.shape[0]
    lapse_features = np.zeros((N, K), dtype=np.float32)
    
    # We try to take eigenvectors 1 to K (0-indexed, so 1..K). 
    # If N <= K, we take what we have.
    
    # Let's stick to the most common practice: Smallest non-trivial eigenvectors.
    # So we skip index 0.
    
    start_idx = 1
    end_idx = start_idx + K
    
    if N > start_idx:
        available = min(N, end_idx) - start_idx
        if available > 0:
            # eigenvectors: (N, N)
            # We take columns start_idx : start_idx + available
            selected_evecs = eigenvectors[:, start_idx : start_idx + available]
            
            # Formula: |normalize(U)|. eigh returns normalized eigenvectors (norm 1).
            # So we just take absolute value.
            lapse_features[:, :available] = np.abs(selected_evecs)
            
    return lapse_features

def compute_entropy_kernel(adj, K):
    G = nx.from_numpy_array(adj)
    entropies = node_khop_entropy_from_nx(G, K=K)
    
    # Compute kernel for each k and average them
    kernel_sum = np.zeros_like(adj, dtype=np.float32)
    
    for k in range(K):
        H_k = entropies[:, k]
        diff = np.abs(H_k[:, None] - H_k[None, :])
        kernel_k = 0.5 * np.exp(-diff)
        kernel_sum += kernel_k
        
    kernel_matrix = kernel_sum / K
    return kernel_matrix

def get_node_kernel_features(G, kernel_matrix, m):
    N = G.number_of_nodes()
    features = np.zeros((N, m), dtype=np.float32)
    
    for node in range(N):
        neighbors = list(G.neighbors(node))
        # Get kernel values for neighbors
        k_values = [kernel_matrix[node, neighbor] for neighbor in neighbors]
        
        # Sort descending to keep strongest connections
        k_values.sort(reverse=True)
        
        if len(k_values) >= m:
            features[node, :] = k_values[:m]
        else:
            features[node, :len(k_values)] = k_values
            # Remaining are 0
            
    return features
