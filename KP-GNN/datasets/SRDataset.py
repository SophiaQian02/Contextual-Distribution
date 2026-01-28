import networkx as nx
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.data import Data
from torch_geometric.utils import to_undirected


class SRDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, se_type="entropy", k=4):
        self.se_type = se_type
        self.k = k
        super(SRDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["sr251256.g6"]  # sr251256  sr351668

    @property
    def processed_file_names(self):
        return f'data_{self.se_type}_{self.k}.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list.
        dataset = nx.read_graph6(self.raw_paths[0])
        data_list = []
        min_node_number = 1e30
        for datum in dataset:
            num_nodes = datum.number_of_nodes()
            if num_nodes < min_node_number:
                min_node_number = num_nodes
        print(min_node_number)
        for i, datum in enumerate(dataset):
            x = torch.ones(datum.number_of_nodes(), 1)
            node_tags = x.squeeze(1)

            edge_index = to_undirected(torch.tensor(list(datum.edges())).transpose(1, 0))
            data_list.append(Data(edge_index=edge_index, x=x, y=0, num_nodes=datum.number_of_nodes(), deg=True, node_tags= node_tags, min_node_number = min_node_number))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def calculate_stats(dataset):
    num_graphs = len(dataset)
    ave_num_nodes = np.array([g.num_nodes for g in dataset]).mean()
    ave_num_edges = np.array([g.num_edges for g in dataset]).mean()
    print(
        f'# Graphs: {num_graphs}, average # nodes per graph: {ave_num_nodes}, average # edges per graph: {ave_num_edges}.')
