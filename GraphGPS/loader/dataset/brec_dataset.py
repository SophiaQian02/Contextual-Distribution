import os
import shutil
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import InMemoryDataset, Data, download_url, extract_zip
from torch_geometric.utils import from_networkx, to_undirected, get_laplacian, to_scipy_sparse_matrix
from tqdm import tqdm
import scipy.sparse.linalg

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

def compute_lapse(edge_index, num_nodes, max_freqs=8, normalization='sym'):
    """
    Compute Laplacian Positional Encoding (LapPE).
    Returns EigVals (N, k), EigVecs (N, k)
    """
    # L = I - D^-1/2 A D^-1/2
    L = to_scipy_sparse_matrix(
        *get_laplacian(edge_index, normalization=normalization, num_nodes=num_nodes)
    )
    
    # Eigen decomposition
    # We need smallest eigenvalues/vectors
    # k = min(num_nodes - 1, max_freqs)
    k = max_freqs + 1
    if num_nodes <= k:
        # Full decomposition if small graph
        evals, evects = np.linalg.eigh(L.toarray())
    else:
        # Partial decomposition
        # 'SM' = Smallest Magnitude
        try:
            evals, evects = scipy.sparse.linalg.eigsh(L, k=k, which='SM', return_eigenvectors=True)
        except:
            # Fallback to full if sparse fails (e.g. disconnected components issues)
            evals, evects = np.linalg.eigh(L.toarray())
            
    # Sort by eigenvalues
    idx = evals.argsort()
    evals = evals[idx]
    evects = evects[:, idx]

    # Filter trivial (remove the first one, assuming connected graph where lambda_1 = 0)
    if len(evals) > 1:
        evals = evals[1:]
        evects = evects[:, 1:]

    # Take absolute value
    evects = np.abs(evects)
    
    # Pad if necessary
    if len(evals) < max_freqs:
        pad_len = max_freqs - len(evals)
        evals = np.pad(evals, (0, pad_len), constant_values=float('nan'))
        evects = np.pad(evects, ((0, 0), (0, pad_len)), constant_values=float('nan'))
    else:
        evals = evals[:max_freqs]
        evects = evects[:, :max_freqs]
        
    return evals, evects

def compute_von_neumann_kernel(edge_index, node_entropies):
    """
    Computes edge kernel features based on node entropies.
    P_uv = exp(-|H(u) - H(v)|)
    """
    row, col = edge_index
    h_u = node_entropies[row] # (E, K)
    h_v = node_entropies[col] # (E, K)
    
    diff = np.abs(h_u - h_v)
    kernel_vals = 0.5 * np.exp(-diff) # (E, K)
    
    # Average over K? Or keep K?
    # GraphGPS KernelPE expects (E, K) usually, or (E, 1) if averaged.
    # Let's keep K dimensions.
    return kernel_vals

class BRECDataset(InMemoryDataset):
    def __init__(
        self,
        root="datasets/BREC",
        name="brec_v3",
        K=3,
        entropy=False,
        rwse=False,
        lapse=False, 
        kernel=False, 
        max_freqs=8,
        laplacian_norm='sym',
        m=10,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.name = name
        self.K = K
        self.entropy = entropy
        self.rwse = rwse
        self.lapse = lapse
        self.kernel = kernel
        self.max_freqs = max_freqs
        self.laplacian_norm = laplacian_norm
        self.m = m
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_dir(self):
        return self.root

    @property
    def processed_dir(self):
        name = "processed"
        if self.entropy:
            name += "_entropy"
        if self.rwse:
            name += "_rwse"
        if self.lapse:
            name += f"_lapse_k{self.max_freqs}_{self.laplacian_norm}"
        if self.kernel:
            name += "_kernel"
            
        if self.entropy or self.rwse or self.kernel:
            name += f"_k{self.K}"
            
        if not self.entropy and not self.rwse and not self.lapse and not self.kernel:
            name += "_no_entropy"
        return os.path.join(self.root, name)

    @property
    def raw_file_names(self):
        return ["brec_v3.npy"]

    @property
    def processed_file_names(self):
        return ["brec_v3.pt"]

    def process(self):
        data_list_raw = np.load(self.raw_paths[0], allow_pickle=True)
        data_list = []
        
        for idx, g_bytes in enumerate(tqdm(data_list_raw, desc="Preprocessing BREC graphs")):
            G = nx.from_graph6_bytes(g_bytes)
            
            # Convert to PyG Data
            data = from_networkx(G)
            
            # Ensure undirected
            if data.edge_index.numel() > 0:
                data.edge_index = to_undirected(data.edge_index)
            
            # Add dummy node features if missing (BREC graphs are structure only)
            if data.x is None:
                data.x = torch.zeros((data.num_nodes, 1), dtype=torch.long)
                
            # Add dummy edge features if missing
            if data.edge_attr is None:
                data.edge_attr = torch.zeros((data.num_edges, 1), dtype=torch.long)

            # Compute Stats
            node_ent = None
            if self.entropy or self.kernel:
                # (N, K)
                node_ent = node_khop_entropy_from_nx(G, K=self.K)
                
            if self.entropy:
                data.pestat_VonNeumann = torch.from_numpy(node_ent).float()
                data.pestat_VonNeumann_k = self.K
            
            if self.kernel:
                # Edge Kernel
                if data.edge_index.numel() > 0:
                    kernel_vals = compute_von_neumann_kernel(data.edge_index.numpy(), node_ent)
                    data.pestat_edge_VonNeumannKernel = torch.from_numpy(kernel_vals).float()
                else:
                    data.pestat_edge_VonNeumannKernel = torch.zeros((0, self.K), dtype=torch.float)

            if self.rwse:
                # (N, K)
                adj = nx.to_numpy_array(G)
                rwse = compute_rwse(adj, K=self.K)
                data.pestat_RWSE = torch.from_numpy(rwse).float()
                
            if self.lapse:
                evals, evects = compute_lapse(data.edge_index, data.num_nodes, max_freqs=self.max_freqs, normalization=self.laplacian_norm)
                data.EigVals = torch.from_numpy(evals).float().unsqueeze(0).repeat(data.num_nodes, 1).unsqueeze(2) # (N, k, 1)
                data.EigVecs = torch.from_numpy(evects).float() # (N, k)
            
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

if __name__ == "__main__":
    dataset = BRECDataset(entropy=True, rwse=True)
    print(len(dataset))
    print(dataset[0])
