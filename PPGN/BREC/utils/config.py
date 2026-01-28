import json
import os
from easydict import EasyDict
import datetime

NUM_CLASSES = {'COLLAB':3, 'IMDBBINARY':2, 'IMDBMULTI':3, 'MUTAG':2, 'NCI1':2, 'NCI109':2, 'PROTEINS':2, 'PTC':2, 'QM9': 12, 'BREC': 16, 'ZINC':1}
NUM_LABELS = {'COLLAB':0, 'IMDBBINARY':0, 'IMDBMULTI':0, 'MUTAG':7, 'NCI1':37, 'NCI109':38, 'PROTEINS':3, 'PTC':22, 'DD':89, 'QM9': 18, 'BREC': 0, 'ZINC':4}
CHOSEN_EPOCH = {'COLLAB': 150, 'IMDBBINARY': 100, 'IMDBMULTI': 150, 'MUTAG': 500, 'NCI1': 200, 'NCI109': 250, 'PROTEINS': 100, 'PTC': 400}
LEARNING_RATES = {'COLLAB': 0.0001, 'IMDBBINARY': 0.00005, 'IMDBMULTI': 0.0001, 'MUTAG': 0.0001, 'NCI1':0.0001, 'NCI109':0.0001, 'PROTEINS': 0.001, 'PTC': 0.0001}
DECAY_RATES = {'COLLAB': 0.5, 'IMDBBINARY': 0.5, 'IMDBMULTI': 0.75, 'MUTAG': 1.0, 'NCI1':0.75, 'NCI109':0.75, 'PROTEINS': 0.5, 'PTC': 1.0}
TIME = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.datetime.now())

def process_config(json_file, dataset_name):    
    
    with open(json_file) as config_file:
        config_dict = json.load(config_file)
    
    config = EasyDict(config_dict)
    #print(config.K)
    
    if dataset_name != '':
        config.dataset_name = dataset_name
        
    config.num_classes = NUM_CLASSES[config.dataset_name]
    
    if config.dataset_name == 'QM9' and config.target_param is not False:
        config.num_classes = 1
        
    if config.dataset_name == 'BREC':
        config.num_classes = 16
        
    if config.dataset_name == "ZINC":
        config.subset = True
        
    config.node_labels = NUM_LABELS[config.dataset_name]
    config.timestamp = TIME
    
    if 'entropy' not in config:
        config.entropy = True
        
    if 'rwse' not in config:
        config.rwse = False
        
    if 'lapse' not in config:
        config.lapse = False
    if "kernel" not in config:
        config.kernel = False
    if "m" not in config:
        config.m = 10
    
    config.parent_dir = config.exp_name + config.dataset_name + TIME
    if "kernel" not in config:
        config.kernel = False
    if "m" not in config:
        config.m = 10
    
    config.summary_dir = os.path.join("../experiments", config.parent_dir, "summary/")
    config.checkpoint_dir = os.path.join("../experiments", config.parent_dir, "checkpoint/")
    if config.exp_name == "10fold_cross_validation":
        config.num_epochs = CHOSEN_EPOCH[config.dataset_name]
        config.hyperparams.learning_rate = LEARNING_RATES[config.dataset_name]
        config.hyperparams.decay_rate = DECAY_RATES[config.dataset_name]
    config.n_gpus = len(config.gpu.split(','))
    config.gpus_list = ",".join(['{}'.format(i) for i in range(config.n_gpus)])
    config.devices = ['/gpu:{}'.format(i) for i in range(config.n_gpus)]
    config.distributed_fold = None
    
    return config
    
    
    
    
if __name__ == '__main__':
    config = process_config('../configs/10fold_config.json', 'MUTAG')
    print(config.values())