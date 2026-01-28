import torch
import data_loader.data_helper as helper
from data_loader.BRECDataset import BRECDataset

class DataGenerator:
    def __init__(self, config):
        self.config = config
        # load data here
        self.batch_size = self.config.hyperparams.batch_size
        self.is_qm9 = self.config.dataset_name == 'QM9'
        self.is_brec = self.config.dataset_name == 'BREC'
        self.is_zinc = self.config.dataset_name == 'ZINC'
        self.labels_dtype = torch.float32 if self.is_zinc or self.is_qm9 or self.is_zinc else torch.long


        self.load_data()
        
    def load_data(self):
        if self.is_qm9:
            self.load_qm9_data()
        elif self.is_brec:
            self.load_brec()
        elif self.is_zinc:
            self.load_zinc()
        else:
            self.load_data_benchmark()
            
        if not self.is_brec:

            self.split_val_test_to_batches()
            
    def load_zinc(self):
        root_dir ="data/ZINC/subset"
        self.train_graphs, self.train_labels = helper.load_zinc_aux(root_dir, split='train', K = self.config.K, entropy=self.config.entropy, rwse=self.config.rwse, lapse=self.config.lapse, kernel=self.config.kernel, m=self.config.m)
        self.val_graphs, self.val_labels = helper.load_zinc_aux(root_dir, split='val', K = self.config.K, entropy=self.config.entropy, rwse=self.config.rwse, lapse=self.config.lapse, kernel=self.config.kernel, m=self.config.m)
        self.test_graphs, self.test_labels = helper.load_zinc_aux(root_dir, split='test',K = self.config.K, entropy=self.config.entropy, rwse=self.config.rwse, lapse=self.config.lapse, kernel=self.config.kernel, m=self.config.m)
        
        self.train_size = len(self.train_graphs)
        self.val_size = len(self.val_graphs)
        self.test_size = len(self.test_graphs)
        
    
    def load_qm9_data(self):
        train_graphs, train_labels, val_graphs, val_labels, test_graphs, test_labels = \
            helper.load_qm9(self.config.target_param, self.config.K, entropy=self.config.entropy, rwse=self.config.rwse, lapse=self.config.lapse, kernel=self.config.kernel, m=self.config.m)
            
        train_labels_mean = train_labels.mean(axis=0)
        train_labels_std = train_labels.std(axis=0)
        train_labels = (train_labels - train_labels_mean) / train_labels_std
        val_labels = (val_labels - train_labels_mean) / train_labels_std
        test_labels = (test_labels - train_labels_mean) / train_labels_std
        
        self.train_graphs, self.train_labels = train_graphs, train_labels
        self.val_graphs, self.val_labels = val_graphs, val_labels
        self.test_graphs, self.test_labels = test_graphs, test_labels
        
        
        self.train_size = len(self.train_graphs)
        self.val_size = len(self.val_graphs)
        self.test_size = len(self.test_graphs)
        self.labels_std = train_labels_std
        
    def load_data_benchmark(self):
        graphs, labels = helper.load_dataset(self.config.dataset_name, self.config.K, entropy=self.config.entropy, rwse=self.config.rwse, lapse=self.config.lapse, kernel=self.config.kernel, m=self.config.m)
        
        
        if self.config.num_fold is None:
            graphs, labels = helper.shuffle(graphs, labels)
            idx = len(graphs) // 10
            self.train_graphs, self.train_labels, self.val_graphs, self.val_labels = graphs[idx:], labels[idx:], graphs[:idx], labels[:idx]
        elif self.config.num_fold == 0:
            train_idx, test_idx = helper.get_parameter_split(self.config.dataset_name)
            self.train_graphs, self.train_labels, self.val_graphs, self.val_labels = graphs[train_idx], labels[
                train_idx], graphs[test_idx], labels[test_idx]
        else:
            train_idx, test_idx = helper.get_train_val_indexes(self.config.num_fold, self.config.dataset_name)
            self.train_graphs, self.train_labels, self.val_graphs, self.val_labels = graphs[train_idx], labels[train_idx], graphs[test_idx], labels[
                test_idx]
            
        self.train_size = len(self.train_graphs)
        self.val_size = len(self.val_graphs)
    
    def load_brec(self):
        self.brec_data = BRECDataset(entropy=self.config.entropy, rwse=self.config.rwse, lapse=self.config.lapse, kernel=self.config.kernel, m=self.config.m)
        
    def get_brec_data(self):
        return self.brec_data
    
        
        
    def next_batch(self):
        graphs, labels = next(self.iter)
        graphs, labels = torch.cuda.FloatTensor(graphs), torch.tensor(labels, device='cuda', dtype=self.labels_dtype)
        return graphs, labels
        
        
    def initialize(self, what_set):
        if what_set == 'train':
            self.reshuffle_data()
        elif what_set == 'val' or what_set == 'validation':
            self.iter = zip(self.val_graphs_batches, self.val_labels_batches)
        elif what_set == 'test':
            self.iter = zip(self.test_graphs_batches, self.test_labels_batches)
        else:
            raise ValueError("what_set should be either 'train', 'val' or 'test'")
        
        
    def reshuffle_data(self):
        """
        Reshuffle train data between epochs
        """
        graphs, labels = helper.group_same_size(self.train_graphs, self.train_labels)
        graphs, labels = helper.shuffle_same_size(graphs, labels)
        graphs, labels = helper.split_to_batches(graphs, labels, self.batch_size)
        self.num_iterations_train = len(graphs)
        graphs, labels = helper.shuffle(graphs, labels)
        self.iter = zip(graphs, labels)
        
        
    def split_val_test_to_batches(self):
        # Split the val and test sets to batchs, no shuffling is needed
        graphs, labels = helper.group_same_size(self.val_graphs, self.val_labels)
        graphs, labels = helper.split_to_batches(graphs, labels, self.batch_size)
        self.num_iterations_val = len(graphs)
        self.val_graphs_batches, self.val_labels_batches = graphs, labels

        if self.is_qm9 or self.is_zinc:
            # Benchmark graphs have no test sets
            graphs, labels = helper.group_same_size(self.test_graphs, self.test_labels)
            graphs, labels = helper.split_to_batches(graphs, labels, self.batch_size)
            self.num_iterations_test = len(graphs)
            self.test_graphs_batches, self.test_labels_batches = graphs, labels
            
        
