import argparse
import os
import time
import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import CosineEmbeddingLoss
from torch_geometric.loader import DataLoader
from torch_geometric import seed_everything
from torch_geometric.graphgym.config import (cfg, load_cfg, set_cfg)
from torch_geometric.graphgym.model_builder import create_model
import graphgps  # Register custom modules
from graphgps.loader.dataset.brec_dataset import BRECDataset

# Setup Logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants from Reference
NUM_RELABEL = 32
P_NORM = 2
OUTPUT_DIM = 16
EPSILON_CMP = 1e-6
SAMPLE_NUM = 400
EPOCH = 20
MARGIN = 0.0
LEARNING_RATE = 1e-4
THRESHOLD = 72.34
BATCH_SIZE = 32
WEIGHT_DECAY = 1e-4
LOSS_THRESHOLD = 0.1

part_dict = {
    "Basic": (0, 60),
    "Regular": (60, 160),
    "Extension": (160, 260),
    "CFI": (260, 360),
    "4-Vertex_Condition": (360, 380),
    "Distance_Regular": (380, 400),
}

def parse_args():
    parser = argparse.ArgumentParser(description='BREC Evaluation')
    parser.add_argument('--cfg', dest='cfg_file', type=str, required=True,
                        help='The configuration file path.')
    parser.add_argument('--dataset', type=str, default='datasets/BREC/brec_v3.npy',
                        help='Path to brec_v3.npy')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='See graphgym/config.py for remaining options.')
    return parser.parse_args()

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, batch):
        out = self.model(batch)
        if isinstance(out, tuple):
            return out[0]
        batch = out
        # Use the prediction head output as embedding
        if hasattr(batch, 'pred'):
            return batch.pred
        elif hasattr(batch, 'graph_feature'):
            return batch.graph_feature
        else:
            raise ValueError("Model does not output pred or graph_feature")

def T2_calculation(model, dataset, device, log_flag=False):
    with torch.no_grad():
        loader = DataLoader(dataset, batch_size=BATCH_SIZE)
        pred_0_list = []
        pred_1_list = []
        for data in loader:
            pred = model(data.to(device)).detach()
            
            pred_0_list.extend(pred[0::2])
            pred_1_list.extend(pred[1::2])
        X = torch.cat([x.reshape(1, -1) for x in pred_0_list], dim=0).T
        Y = torch.cat([x.reshape(1, -1) for x in pred_1_list], dim=0).T
        if log_flag:
            logger.info(f"X_mean = {torch.mean(X, dim=1)}")
            logger.info(f"Y_mean = {torch.mean(Y, dim=1)}")

        D = X - Y
        D_mean = torch.mean(D, dim=1).reshape(-1, 1)
        
        S = torch.cov(D)

        inv_S = torch.linalg.pinv(S)
        return torch.mm(torch.mm(D_mean.T, inv_S), D_mean).item()
        
        # Numerical stability
        try:
            inv_S = torch.linalg.pinv(S)
            t2 = torch.mm(torch.mm(D_mean.T, inv_S), D_mean).item()
        except:
            t2 = 0.0
        return t2

def main():
    args = parse_args()
    
    # Load Config
    set_cfg(cfg)
    load_cfg(cfg, args)
    
    # Parse kernel times_func for PEs (Critical for RWSE)
    for key, pecfg in cfg.items():
        if key.startswith('posenc_') and pecfg.enable:
            if hasattr(pecfg, 'kernel'):
                if pecfg.kernel.times_func:
                    pecfg.kernel.times = list(eval(pecfg.kernel.times_func))
    
    # Set device
    device = torch.device(args.device)
    cfg.accelerator = args.device
    
    # Force output dim for BREC
    # GraphGym uses cfg.share.num_classes or cfg.dataset.num_classes
    # We need to ensure the model head outputs OUTPUT_DIM
    # For SANGraphHead, it uses cfg.dataset.num_classes (usually inferred)
    # We override it here.
    cfg.dataset.num_classes = OUTPUT_DIM
    # Also update share.dim_out if it exists, as GraphGym might have pre-calculated it
    if hasattr(cfg, 'share'):
        cfg.share.dim_out = OUTPUT_DIM
        cfg.share.num_classes = OUTPUT_DIM
    
    # Determine enabled PEs from cfg for logging filename
    pe_list = []
    if getattr(cfg.posenc_RWSE, 'enable', False): pe_list.append("RWSE")
    if getattr(cfg.posenc_LapPE, 'enable', False): pe_list.append("LapPE")
    if getattr(cfg.posenc_VonNeumann, 'enable', False): pe_list.append("Von")
    if getattr(cfg.posenc_VonNeumannKernel, 'enable', False): pe_list.append("VonKernel")
    pe_tag = "+".join(pe_list) if pe_list else "NoPE"
    
    if not os.path.exists('results'):
        os.makedirs('results')
    filename = f"results/BREC_{pe_tag}.txt"
    
    # Setup File Logging
    fh = logging.FileHandler(filename, mode='w') # Overwrite mode initially
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # Load Dataset
    root_dir = os.path.dirname(args.dataset)
    
    # Determine enabled PEs from cfg
    entropy = getattr(cfg.posenc_VonNeumann, 'enable', False)
    rwse = getattr(cfg.posenc_RWSE, 'enable', False)
    lapse = getattr(cfg.posenc_LapPE, 'enable', False)
    kernel = getattr(cfg.posenc_VonNeumannKernel, 'enable', False)
    
    # Determine K
    K = 3
    if entropy:
        K = getattr(cfg.posenc_VonNeumann, 'k_order', 3)
    elif kernel:
        K = getattr(cfg.posenc_VonNeumannKernel, 'k_order', 3)
    elif rwse:
        K = len(cfg.posenc_RWSE.kernel.times)

    # Determine LapPE params
    max_freqs = 8
    laplacian_norm = 'sym'
    if lapse:
        max_freqs = cfg.posenc_LapPE.eigen.max_freqs
        laplacian_norm = cfg.posenc_LapPE.eigen.laplacian_norm
        if laplacian_norm == 'none':
            laplacian_norm = None

    logger.info(f"Loading BREC dataset from {root_dir}...")
    dataset = BRECDataset(root=root_dir, 
                          entropy=entropy,
                          rwse=rwse,
                          lapse=lapse,
                          kernel=kernel,
                          K=K,
                          max_freqs=max_freqs,
                          laplacian_norm=laplacian_norm)
    
    logger.info(f"Dataset loaded. Total graphs: {len(dataset)}")
    
    # Evaluation Loop
    cnt = 0
    correct_list = []
    fail_in_reliability = 0
    loss_func = CosineEmbeddingLoss(margin=MARGIN)
    
    results = {}
    
    time_start = time.process_time()
    
    for part_name, part_range in part_dict.items():
        logger.info(f"{part_name} part starting ---")
        
        cnt_part = 0
        fail_in_reliability_part = 0
        y = torch.tensor([-1] * (BATCH_SIZE // 2)).to(device)
        
        start = time.process_time()
        
        for id in range(part_range[0], part_range[1]):
            logger.info(f"ID: {id}")
            
            # Re-initialize model for each pair
            seed_everything(cfg.seed) # Reset seed to ensure consistent init if needed
            base_model = create_model()
            base_model.to(device)
            model = ModelWrapper(base_model)
            
            optimizer = torch.optim.Adam(
                model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
            
            # Select data slices
            # Test Pair
            idx_start = id * NUM_RELABEL * 2
            idx_end = (id + 1) * NUM_RELABEL * 2
            dataset_traintest = dataset[idx_start : idx_end]
            
            # Reliability Pair
            idx_rel_start = (id + SAMPLE_NUM) * NUM_RELABEL * 2
            idx_rel_end = (id + SAMPLE_NUM + 1) * NUM_RELABEL * 2
            dataset_reliability = dataset[idx_rel_start : idx_rel_end]
            
            # Train
            model.train()
            for _ in range(EPOCH):
                traintest_loader = DataLoader(dataset_traintest, batch_size=BATCH_SIZE, shuffle=False)
                loss_all = 0
                for data in traintest_loader:
                    optimizer.zero_grad()
                    pred = model(data.to(device))
                    # pred is (Batch, Dim)
                    # We assume batch is interleaved: G1, G2, G1, G2...
                    # pred[0::2] are G1s, pred[1::2] are G2s
                    
                    # Handle last batch if size mismatch
                    current_batch_size = pred.size(0)
                    if current_batch_size % 2 != 0:
                        # Should not happen if NUM_RELABEL * 2 is multiple of BATCH_SIZE
                        # NUM_RELABEL=32, *2=64. BATCH_SIZE=32. 
                        # So 2 batches of 32. Perfect.
                        continue
                        
                    target = y
                    if current_batch_size // 2 != target.size(0):
                        target = torch.tensor([-1] * (current_batch_size // 2)).to(device)
                        
                    loss = loss_func(pred[0::2], pred[1::2], target)
                    loss.backward()
                    optimizer.step()
                    loss_all += (current_batch_size / 2) * loss.item()
                
                loss_all /= NUM_RELABEL
                logger.info(f"Loss: {loss_all}")
                if loss_all < LOSS_THRESHOLD:
                    # logger.info("Early Stop Here")
                    break
                scheduler.step(loss_all)
            
            # Eval
            model.eval()
            T_square_traintest = T2_calculation(model, dataset_traintest, device, True)
            T_square_reliability = T2_calculation(model, dataset_reliability, device, True)
            
            isomorphic_flag = False
            reliability_flag = False
            

            
            if T_square_traintest > THRESHOLD and not torch.isclose(
                torch.tensor(T_square_traintest), torch.tensor(T_square_reliability), atol=EPSILON_CMP
            ):
                isomorphic_flag = True
            
            if T_square_reliability < THRESHOLD:
                reliability_flag = True
                
            if isomorphic_flag:
                cnt += 1
                cnt_part += 1
                correct_list.append(id)
            
            if not reliability_flag:
                fail_in_reliability += 1
                fail_in_reliability_part += 1
                
            logger.info(f"ID {id}: Iso={isomorphic_flag} (T2={T_square_traintest:.2f}), Rel={reliability_flag} (T2={T_square_reliability:.2f})")
            
        end = time.process_time()
        time_cost_part = round(end - start, 2)
        
        acc_part = cnt_part / (part_range[1] - part_range[0])
        results[part_name] = acc_part
        
        logger.info(
            f"{part_name} part costs time {time_cost_part}; Correct in {cnt_part} / {part_range[1] - part_range[0]}"
        )
        logger.info(
            f"Fail in reliability: {fail_in_reliability_part} / {part_range[1] - part_range[0]}"
        )

    time_end = time.process_time()
    time_cost = round(time_end - time_start, 2)
    logger.info(f"evaluation time cost: {time_cost}")

    Acc = round(cnt / SAMPLE_NUM, 2)
    logger.info(f"Correct in {cnt} / {SAMPLE_NUM}, Acc = {Acc}")
    logger.info(f"Fail in reliability: {fail_in_reliability} / {SAMPLE_NUM}")
    logger.info(correct_list)
    
    # Save results summary (append to log file)
    with open(filename, 'a') as f:
        f.write(f"\n{'='*20} Final Results {'='*20}\n")
        f.write(f"Configuration:\n")
        f.write(f"PE: {pe_tag}\n")
        f.write(f"Seed: {cfg.seed}\n")
        f.write(f"Total Acc: {Acc}\n")
        f.write("-" * 20 + "\n")
        for cat, acc in results.items():
            f.write(f"{cat}: {acc:.4f}\n")
            
    logger.info(f"Results saved to {filename}")

if __name__ == '__main__':
    main()
