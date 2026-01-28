"""
EXP dataset
"""
import os
import pickle

import torch
from torch_geometric.data import InMemoryDataset, Data


# NAME = "GRAPHSAT"
class PlanarSATPairsDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, se_type="entropy", k=4):
        self.se_type = se_type
        self.k = k
        super(PlanarSATPairsDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["GRAPHSAT.pkl"]

    @property
    def processed_file_names(self):
        return f'data_{self.se_type}_{self.k}.pt'

    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        data_list = pickle.load(open(os.path.join(self.root, "raw/GRAPHSAT.pkl"), "rb"))
        
        min_node_number = 1e30
        for g in data_list:
            x = g.__dict__['x']
            num_nodes = x.shape[0]
            if min_node_number > num_nodes:
                min_node_number = num_nodes
        print(min_node_number)
        new_data_list = []
        for g in data_list:
            edge_index = g.__dict__['edge_index']
            y = g.__dict__['y']
            x = g.__dict__['x']
            num_nodes = x.shape[0]
            node_tags = x .squeeze(1)
            data = Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes, deg=False, node_tags = node_tags, min_node_number=min_node_number)
            new_data_list.append(data)
            
            
            
        
        data_list = new_data_list
        
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


if __name__ == "__main__":
    test_path = "../data/EXP/"
    dataset = PlanarSATPairsDataset(test_path)
    print(dataset[0])
