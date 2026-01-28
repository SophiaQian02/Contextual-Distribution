import os
import pickle
import numpy as np
import sys
from tqdm import tqdm
import os.path as osp
import torch


project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)) 

sys.path.append(project_dir) 
os.chdir(project_dir) 
from data_loader.data_utils import node_khop_entropy_from_nx, compute_rwse, compute_lapse, compute_entropy_kernel, get_node_kernel_features
import networkx as nx

NUM_LABELS = {'ENZYMES': 3, 'COLLAB': 0, 'IMDBBINARY': 0, 'IMDBMULTI': 0, 'MUTAG': 7, 'NCI1': 37, 'NCI109': 38,
              'PROTEINS': 3, 'PTC': 22, 'DD': 89}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))



def load_dataset(ds_name, K=3, entropy=True, rwse=False, lapse=False, kernel=False, m=10):
    """
    construct graphs and labels from dataset text in data folder
    :param ds_name: name of data set you want to load
    :return: two numpy arrays of shape (num_of_graphs).
            the graphs array contains in each entry a ndarray represent adjacency matrix of a graph of shape (num_vertex, num_vertex, num_vertex_labels)
            the labels array in index i represent the class of graphs[i]
    """
    directory = BASE_DIR + "/data/benchmark_graphs/{0}/{0}.txt".format(ds_name)
    graphs = []
    labels = []
    with open(directory, "r") as data:
        num_graphs = int(data.readline().rstrip().split(" ")[0])
        for i in range(num_graphs):
            graph_meta = data.readline().rstrip().split(" ")
            num_vertex = int(graph_meta[0])
            
            extra_channels = 0
            if entropy:
                extra_channels += K
            if rwse:
                extra_channels += K
            if lapse:
                extra_channels += K
            if kernel:
                extra_channels += m
                
                
            num_channels = NUM_LABELS[ds_name]+1 + extra_channels
            curr_graph = np.zeros(shape=(num_vertex, num_vertex, num_channels), dtype=np.float32)
            labels.append(int(graph_meta[1]))
            adj = np.zeros((num_vertex, num_vertex), dtype=np.float32)
            
            vertex_lines = []
            for j in range(num_vertex):
                vertex_lines.append(data.readline().rstrip().split(" "))
                
            # Build adjacency
            for j, vertex in enumerate(vertex_lines):
                # vertex[0] is label
                # vertex[1] is number of neighbors? No, usually it is just neighbors after label.
                # The code says: `for k in range(2,len(vertex)): curr_graph[j, int(vertex[k]), 0] = 1.`
                # So vertex[0] is label. vertex[1] is maybe number of neighbors? Or just skipped?
                # Let's assume the existing code logic for parsing is correct about indices.
                
                if NUM_LABELS[ds_name] != 0:
                    curr_graph[j, j, int(vertex[0])+1]= 1.
                for k in range(2,len(vertex)):
                    curr_graph[j, int(vertex[k]), 0] = 1.
                    adj[j, int(vertex[k])] = 1.
                    adj[int(vertex[k]), j] = 1. # Undirected?
                    
            # Now compute features
            if entropy or rwse or lapse or kernel:
                # Re-create G from populated adj
                G = nx.from_numpy_array(adj)
                
            if entropy:
                ent_matrix = node_khop_entropy_from_nx(G, K=K)
                
            if rwse:
                rwse_matrix = compute_rwse(adj, K=K)
                
            if lapse:
                lapse_matrix = compute_lapse(adj, K=K)
                
            if kernel:
                kernel_matrix = compute_entropy_kernel(adj, K=K)
                kernel_feats = get_node_kernel_features(G, kernel_matrix, m=m)
            
            # Assign features
            current_channel_idx = 1 + NUM_LABELS[ds_name]
            
            if entropy:        
                for k in range(K):
                    curr_graph[:, :, current_channel_idx + k] = np.diag(ent_matrix[:, k])
                current_channel_idx += K
            if rwse:
                for k in range(K):
                    curr_graph[:, :, current_channel_idx + k] = np.diag(rwse_matrix[:, k])
                current_channel_idx += K
            if lapse:
                for k in range(K):
                    curr_graph[:, :, current_channel_idx + k] = np.diag(lapse_matrix[:, k])
                current_channel_idx += K
            if kernel:
                for k in range(m):
                    curr_graph[:, :, current_channel_idx + k] = np.diag(kernel_feats[:, k])
                current_channel_idx += m
            curr_graph = normalize_graph(curr_graph)
            graphs.append(curr_graph)
            
    # ...
    # for i, graph in enumerate(graphs):
    #     print(f"Shape of graph {i}: {graph.shape}")
    # for i in range(len(graphs)):
    #     graphs[i] = np.transpose(graphs[i], [2, 0, 1])
    
    graphs = np.array(graphs, dtype=object)
    for i in range(len(graphs)):
        graphs[i] = np.transpose(graphs[i], [2, 0, 1]) 

    labels = np.array(labels)
    _, labels = np.unique(labels, return_inverse=True)
    
    #print(len(graphs), labels.shape)
    
   
    return graphs, labels


def load_qm9(target_param, K, entropy=True, rwse=False, lapse=False, kernel=False, m=10):
    """
    Constructs the graphs and labels of QM9 data set, already split to train, val and test sets
    :return: 6 numpy arrays:
                 train_graphs: N_train,
                 train_labels: N_train x 12, (or Nx1 is target_param is not False)
                 val_graphs: N_val,
                 val_labels: N_train x 12, (or Nx1 is target_param is not False)
                 test_graphs: N_test,
                 test_labels: N_test x 12, (or Nx1 is target_param is not False)
                 each graph of shape: 19 x Nodes x Nodes (CHW representation)
    """
    train_graphs, train_labels = load_qm9_aux('train', target_param, K, entropy, rwse, lapse, kernel, m)
    val_graphs, val_labels = load_qm9_aux('val', target_param, K, entropy, rwse, lapse, kernel, m)
    test_graphs, test_labels = load_qm9_aux('test', target_param, K, entropy, rwse, lapse, kernel, m)
    return train_graphs, train_labels, val_graphs, val_labels, test_graphs, test_labels


# def load_qm9_aux(which_set, target_param):
#     """
#     Read and construct the graphs and labels of QM9 data set, already split to train, val and test sets
#     :param which_set: 'test', 'train' or 'val'
#     :param target_param: if not false, return the labels for this specific param only
#     :return: graphs: (N,)
#              labels: N x 12, (or Nx1 is target_param is not False)
#              each graph of shape: 19 x Nodes x Nodes (CHW representation)
#     """
#     base_path = BASE_DIR + "/data/QM9/raw/QM9_{}.p".format(which_set)
#     graphs, labels = [], []
#     with open(base_path, 'rb') as f:
#         data = pickle.load(f)
#         for instance in data:
#             y = instance['y']
            
#             # Ensure label is 12-dimensional (take first 12 if longer)
#             if y.shape[1] > 12:
#                 y = y[:,:12]
#             labels.append(y)
#             nodes_num = instance['usable_features']['x'].shape[0]
#             graph = np.empty((nodes_num, nodes_num, 17))
#             for i in range(11):
#                 # 13 features per node - for each, create a diag matrix of it as a feature
#                 graph[:, :, i] = np.diag(instance['usable_features']['x'][:, i])
#             graph[:, :, 11] = instance['usable_features']['distance_mat']
#             graph[:, :, 12] = instance['usable_features']['affinity']
#             graph[:, :, 13:] = instance['usable_features']['edge_features']  # shape n x n x 4
#             graphs.append(graph)
#     graphs = np.array(graphs, dtype=object)
#     for i in range(graphs.shape[0]):
#         graphs[i] = np.transpose(graphs[i], [2, 0, 1])
#     labels = np.array(labels).squeeze()  # shape N x 12
#     if target_param is not False:  # regression over a specific target, not all 12 elements
#         labels = labels[:, target_param].reshape(-1, 1)  # shape N x 1

#     return graphs, labels

def load_qm9_aux(which_set, target_param, K, entropy=True, rwse=False, lapse=False, kernel=False, m=10):
    """
    Load QM9 with K k-hop entropy diagonal channels (total 17 + K channels).
    Channels:
      0–10: node features (11)
      11: distance matrix
      12: affinity (adjacency)
      13–16: edge features (4)
      17 to 16+K: k=1 to K k-hop entropies (diagonal)
    """
    import networkx as nx
    base_path = os.path.join(BASE_DIR, "data/QM9/raw", f"QM9_{which_set}.p")
    graphs, labels = [], []

    with open(base_path, 'rb') as f:
        data = pickle.load(f)
        for instance in tqdm(data, desc=f"Loading QM9 ({which_set})", unit="mol"):
            y = instance['y']
            if y.ndim == 2 and y.shape[1] > 12:
                y = y[:, :12]
            labels.append(y)

            uf = instance['usable_features']
            x = uf['x']  # (N, 11)
            n = x.shape[0]

            # Build NetworkX graph from affinity (adjacency)
            A = uf['affinity']  # (N, N)
            G = nx.from_numpy_array(A)

            # Compute k-hop entropies for k=1 to K → shape (N, K)
            if entropy:
                ent_matrix = node_khop_entropy_from_nx(G, K=K)  # (N, K)
                
            if rwse:
                rwse_matrix = compute_rwse(A, K=K)
                
            if lapse:
                lapse_matrix = compute_lapse(A, K=K)

            if kernel:
                kernel_matrix = compute_entropy_kernel(A, K=K)
                kernel_feats = get_node_kernel_features(G, kernel_matrix, m=m)

            # Total channels = 17 (original) + K (entropy)
            extra_channels = 0
            if entropy:
                extra_channels += K
            if rwse:
                extra_channels += K
            if lapse:
                extra_channels += K
            if kernel:
                extra_channels += m
                
                
            total_channels = 17 + extra_channels
            graph = np.zeros((n, n, total_channels), dtype=np.float32)

            # Channels 0–10: node features (11 dims)
            for i in range(min(11, x.shape[1])):
                graph[:, :, i] = np.diag(x[:, i])

            # Channel 11: distance
            graph[:, :, 11] = uf['distance_mat']

            # Channel 12: affinity
            graph[:, :, 12] = A

            # Channels 13–16: edge features (4 dims)
            graph[:, :, 13:17] = uf['edge_features']

            current_channel_idx = 17
            if entropy:
                for k in range(K):
                    graph[:, :, current_channel_idx + k] = np.diag(ent_matrix[:, k])
                current_channel_idx += K
            if rwse:
                for k in range(K):
                    graph[:, :, current_channel_idx + k] = np.diag(rwse_matrix[:, k])
                current_channel_idx += K
            if lapse:
                for k in range(K):
                    graph[:, :, current_channel_idx + k] = np.diag(lapse_matrix[:, k])
                current_channel_idx += K
            if kernel:
                for k in range(m):
                    graph[:, :, current_channel_idx + k] = np.diag(kernel_feats[:, k])
                current_channel_idx += m
            
            graph = np.transpose(graph, (2, 0, 1))
            graphs.append(graph)

    labels = np.array(labels).squeeze()
    if target_param is not False:
        labels = labels[:, target_param].reshape(-1, 1)

    return graphs, labels


def load_zinc_aux(processed_dir, split='train', K=3, entropy=True, rwse=False, lapse=False, kernel=False, m=10):
    path = osp.join(processed_dir+"/processed", f'{split}.pt')
    if not osp.exists(path):
        raise FileNotFoundError(f"Processed data for {split} not found at {path}")

    data, slices = torch.load(path, map_location='cpu')
    

    max_nodes = 38  
    graphs = []
    labels = []

    
    num_graphs = slices['y'].size(0)-1 if 'y' in slices else len(slices['x']) - 1
    print(num_graphs)
    for i in tqdm(range(num_graphs),
                  desc=f"Preprocessing ZINC ({split})",
                  ncols=80):
        
        start = slices['x'][i]
        end = slices['x'][i + 1]
        x = data.x[start:end].view(-1).numpy()  # (n,)

        
        edge_start = slices['edge_index'][i]
        edge_end = slices['edge_index'][i + 1]
        edge_index = data.edge_index[:, edge_start:edge_end].numpy()  # (2, E)
        edge_attr = data.edge_attr[edge_start:edge_end].numpy()       # (E,)

        y = data.y[i].item()

        n = x.shape[0]
        #print(x.shape)
        assert n <= max_nodes

        
        node_mat = np.zeros((max_nodes, max_nodes), dtype=np.float32)
        node_mat[np.arange(n), np.arange(n)] = x

        
        edge_dense = np.zeros((4, max_nodes, max_nodes), dtype=np.float32)
        A = np.zeros((n, n), dtype=np.float32)
        for j in range(edge_index.shape[1]):
            src, dst = edge_index[:, j]
            typ = edge_attr[j]
            A[src, dst] = 1.0
            A[dst, src] = 1.0
            if 1 <= typ <= 4:
                ch = typ - 1
                edge_dense[ch, src, dst] = 1.0
                edge_dense[ch, dst, src] = 1.0  
        
        parts = [node_mat[np.newaxis, :, :], edge_dense]

        if entropy and K > 0:
            entropy_dense = np.zeros((K, max_nodes, max_nodes), dtype=np.float32)
            G = nx.from_numpy_array(A)               
            ent_matrix = node_khop_entropy_from_nx(G, K=K)   # (n, K)
            
            for k in range(K):
                
                entropy_dense[k, np.arange(n), np.arange(n)] = ent_matrix[:, k]
            parts.append(entropy_dense)
            
        if rwse and K > 0:
            rwse_dense = np.zeros((K, max_nodes, max_nodes), dtype=np.float32)
            rwse_matrix = compute_rwse(A, K=K)
            for k in range(K):
                rwse_dense[k, np.arange(n), np.arange(n)] = rwse_matrix[:, k]
            parts.append(rwse_dense)
            
        if lapse and K > 0:
            lapse_dense = np.zeros((K, max_nodes, max_nodes), dtype=np.float32)
            lapse_matrix = compute_lapse(A, K=K)
            for k in range(K):
                lapse_dense[k, np.arange(n), np.arange(n)] = lapse_matrix[:, k]
            parts.append(lapse_dense)

        if kernel and K > 0:
            kernel_dense = np.zeros((m, max_nodes, max_nodes), dtype=np.float32)
            # Recompute G if needed (it is needed if entropy wasn't called)
            # But G was created inside `if entropy`. So we might need to create it here.
            # But wait, A is available.
            
            kernel_matrix = compute_entropy_kernel(A, K=K)
            # We need G for get_node_kernel_features
            G = nx.from_numpy_array(A)
            kernel_feats = get_node_kernel_features(G, kernel_matrix, m=m)
            
            for k in range(m):
                kernel_dense[k, np.arange(n), np.arange(n)] = kernel_feats[:, k]
            parts.append(kernel_dense)

        graph = np.concatenate(parts, axis=0)
        graphs.append(graph)
        labels.append(y)

    graphs = np.stack(graphs, axis=0)        # (N, 5+K, 38, 38)
    labels = np.array(labels).reshape(-1, 1)  # (N, 1)
    #print(labels.shape)

    return graphs, labels

def shuffle(graphs, labels):
    shf = np.arange(labels.shape[0], dtype=np.int32)
    np.random.shuffle(shf)
    return np.array(graphs)[shf], labels[shf]


def get_parameter_split(ds_name):
    """
    reads the indexes of a specific split to train and validation sets from data folder
    :param ds_name: name of data set
    :return: indexes of the train and test graphs
    """
    directory = BASE_DIR + "/data/benchmark_graphs/{0}/".format(ds_name)
    train_file = "tests_train_split.txt"
    train_idx=[]
    with open(os.path.join(directory, train_file), 'r') as file:
        for line in file:
            train_idx.append(int(line.rstrip()))
    test_file = "tests_val_split.txt"
    test_idx = []
    with open(os.path.join(directory, test_file), 'r') as file:
        for line in file:
            test_idx.append(int(line.rstrip()))
    return train_idx, test_idx


def get_train_val_indexes(num_val, ds_name):
    """
    reads the indexes of a specific split to train and validation sets from data folder
    :param num_val: number of the split
    :param ds_name: name of data set
    :return: indexes of the train and test graphs
    """
    directory = BASE_DIR + "/data/benchmark_graphs/{0}/10fold_idx".format(ds_name)
    train_file = "train_idx-{0}.txt".format(num_val)
    train_idx=[]
    with open(os.path.join(directory, train_file), 'r') as file:
        for line in file:
            train_idx.append(int(line.rstrip()))
    test_file = "test_idx-{0}.txt".format(num_val)
    test_idx = []
    with open(os.path.join(directory, test_file), 'r') as file:
        for line in file:
            test_idx.append(int(line.rstrip()))
    return train_idx, test_idx


def group_same_size(graphs, labels):
    
    # t : N*N*Channel
    sizes = list(map(lambda t: t.shape[1], graphs)) 
    
    indexes = np.argsort(sizes)
    graphs = graphs[indexes]
    labels = labels[indexes]
    r_graphs = []
    r_labels = []
    one_size = []
    start = 0
    size = graphs[0].shape[1]
    for i in range(len(graphs)):
        if graphs[i].shape[1] == size:
            #print(np.expand_dims(graphs[i], axis=0))
            one_size.append(np.expand_dims(graphs[i], axis=0))
        else:
            r_graphs.append(np.concatenate(one_size, axis=0))
            r_labels.append(np.array(labels[start:i]))
            start = i
            one_size = []
            size = graphs[i].shape[1]
            one_size.append(np.expand_dims(graphs[i], axis=0))
    r_graphs.append(np.concatenate(one_size, axis=0))
    r_labels.append(np.array(labels[start:]))
    
    return r_graphs, r_labels


def normalize_graph(curr_graph):

    split = np.split(curr_graph, [1], axis=2)

    adj = np.squeeze(split[0], axis=2)
    deg = np.sqrt(np.sum(adj, 0))
    deg = np.divide(1., deg, out=np.zeros_like(deg), where=deg!=0)
    normal = np.diag(deg)
    norm_adj = np.expand_dims(np.matmul(np.matmul(normal, adj), normal), axis=2)
    ones = np.ones(shape=(curr_graph.shape[0], curr_graph.shape[1], curr_graph.shape[2]), dtype=np.float32)
    spred_adj = np.multiply(ones, norm_adj)
    labels= np.append(np.zeros(shape=(curr_graph.shape[0], curr_graph.shape[1], 1)), split[1], axis=2)
    return np.add(spred_adj, labels)



def split_to_batches(graphs, labels, size):
    """
    split the same size graphs array to batches of specified size
    last batch is in size num_of_graphs_this_size % size
    :param graphs: array of arrays of same size graphs
    :param labels: the corresponding labels of the graphs
    :param size: batch size
    :return: two arrays. graphs array of arrays in size (batch, num vertex, num vertex. num vertex labels)
                corresponds labels
    """
    r_graphs = []
    r_labels = []
    for k in range(len(graphs)):
        r_graphs = r_graphs + np.split(graphs[k], [j for j in range(size, graphs[k].shape[0], size)])
        r_labels = r_labels + np.split(labels[k], [j for j in range(size, labels[k].shape[0], size)])

    # Avoid bug for batch_size=1, where instead of creating numpy array of objects, we had numpy array of floats with
    # different sizes - could not reshape
    ret1, ret2 = np.empty(len(r_graphs), dtype=object), np.empty(len(r_labels), dtype=object)
    ret1[:] = r_graphs
    ret2[:] = r_labels
    return ret1, ret2


def shuffle_same_size(graphs, labels):
    r_graphs, r_labels = [], []
    for i in range(len(labels)):
        curr_graph, curr_labels = shuffle(graphs[i], labels[i])
        r_graphs.append(curr_graph)
        r_labels.append(curr_labels)
    return r_graphs, r_labels
