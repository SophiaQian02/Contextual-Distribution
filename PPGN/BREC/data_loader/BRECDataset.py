import networkx as nx
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import (
    InMemoryDataset,
    download_url,
    extract_zip,
)
from torch_geometric.utils.convert import from_networkx
import os
from tqdm import tqdm
import shutil
from data_loader.data_utils import compute_lapse, compute_entropy_kernel, get_node_kernel_features, node_khop_entropy_from_nx, compute_rwse

torch_geometric.seed_everything(2022)


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



def graph6_to_pyg(x):
    return from_networkx(nx.from_graph6_bytes(x))


def add_x(data):
    data.x = torch.ones([data.num_nodes, 1])
    return data


class BRECDataset(InMemoryDataset):
    url = "https://www.dropbox.com/scl/fi/lzbll513dcgo9x9d3ozso/brec_v3.zip?rlkey=pyd7yss25wasajyffu47ierg3&dl=1"

    def __init__(
        self,
        K = 3,
        entropy = True,
        rwse = False,
        lapse = False, kernel = False, m = 10,
        name="no_param",
        root="data/brec_v3",
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        """_summary_

        Args:
            name (str, optional): The name of processed datasets if with additional settings for data processing. Defaults to "no_param".
            root (str, optional): Data file directory. Defaults to "Data".
            transform (_type_, optional): Data transform function. Defaults to None.
            pre_transform (_type_, optional): Data pre-transform function. Defaults to None.
            pre_filter (_type_, optional): Data pre-filter function. Defaults to None.
        """
        self.root = root
        self.name = name
        self.K = K
        self.entropy = entropy
        self.rwse = rwse
        self.lapse = lapse
        self.kernel = kernel
        self.m = m
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data = torch.load(self.processed_paths[0])

    @property
    def processed_dir(self):
        name = "processed"
        if self.entropy:
            name += "_entropy"
        if self.rwse:
            name += "_rwse"
        if self.lapse:
            name += "_lapse"
        if self.kernel:
            name += "_kernel"
            
        if not self.entropy and not self.rwse and not self.lapse and not self.kernel:
            name += "_no_entropy"
        return os.path.join(self.root, self.name, name)

    @property
    def raw_file_names(self):
        return ["brec_v3.npy"]

    @property
    def processed_file_names(self):
        return ["brec_v3.pt"]

    # def download(self):
    #     shutil.rmtree(self.raw_dir)
    #     path = download_url(self.url, self.root)
    #     extract_zip(path, self.raw_dir)
    #     os.unlink(path)

    def process(self):
        data_list = np.load(self.raw_paths[0], allow_pickle=True)
        data = []
        for idx, g in enumerate(tqdm(data_list, desc="Preprocessing BREC graphs")):
            g_networkx = nx.from_graph6_bytes(g)
            G = g_networkx
            N = G.number_of_nodes()
            A = nx.to_numpy_array(G) 
            A_with_self = A + np.eye(N)
            channels = [A_with_self]
            if self.entropy:
                node_ent = node_khop_entropy_from_nx(G, K=self.K)
                for k in range(self.K):
                    diag_k = np.diag(node_ent[:, k])
                    channels.append(diag_k)
            
            if self.rwse:
                rwse_matrix = compute_rwse(A, K=self.K)
                for k in range(self.K):
                    diag_k = np.diag(rwse_matrix[:, k])
                    channels.append(diag_k)
                    
            if self.lapse: 
                lapse_matrix = compute_lapse(A, K=self.K)
                for k in range(self.K):
                    diag_k = np.diag(lapse_matrix[:, k])
                    channels.append(diag_k)
                    
            if self.kernel:
                kernel_matrix = compute_entropy_kernel(A, K=self.K)
                kernel_feats = get_node_kernel_features(G, kernel_matrix, m=self.m)
                for k in range(self.m):
                    diag_k = np.diag(kernel_feats[:, k])
                    channels.append(diag_k)
            graph_dense = np.stack(channels, axis=0)
            # ent_k = node_ent[:, self.K - 1]                    # shape (N,)
            # diag_k = np.diag(ent_k)
            # diag_k = diag_k 
            # channels.append(diag_k)
            # graph_dense = np.stack(channels, axis=0)
            #graph_dense= np.expand_dims(diag_k, axis=0)
            
            data.append(torch.tensor(graph_dense, dtype=torch.float32))
            # edge = np.expand_dims(nx.to_numpy_array(g_networkx), axis=0)
            # node = np.expand_dims(np.eye(g_networkx.number_of_nodes()), axis=0)
            #data.append(torch.tensor(edge + node, dtype=torch.float32))
            # data.append(
            #     torch.tensor(
            #         np.concatenate((edge, node), axis=0),
            #         dtype=torch.float32,
            #     )
            # )
        self.data = data

        torch.save(self.data, self.processed_paths[0])
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        return data


def main():
    dataset = BRECDataset()
    print(len(dataset))


if __name__ == "__main__":
    main()
