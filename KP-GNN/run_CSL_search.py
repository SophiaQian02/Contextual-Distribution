import os
import argparse
from itertools import product

parser = argparse.ArgumentParser(f'add multi-gpu option')
parser.add_argument("--parallel", action="store_true",
                    help="If true, use DataParallel for multi-gpu training")
parser.add_argument('--se_type', type=str, default="entropy", choices=("entropy", "RWSE", "LapSE"),
                    help="The type of structural encoding to use")
parallel_arg = parser.parse_args()
parallel = parallel_arg.parallel
se_type = parallel_arg.se_type
ks = [4]
kernels = ["gd"]
models = ["KPGIN"]
grid = product(kernels, ks, models)

if parallel:                                                                                                                                                                             
    for parameters in grid:
        kernel, k, model = parameters
        # KP-GNN
        script = f"python3 train_CSL.py --model_name={model} --kernel={kernel} --K={k}  --num_layer=2 --reprocess  --parallel  --se_type={se_type}"
        os.system(script)
        # K-hop GNN
        # script = f"python train_CSL.py --kernel={kernel} --K={k} --num_layer=2 --wo_peripheral_edge --wo_peripheral_configuration --wo_path_encoding --reprocess  --parallel"
        # os.system(script)
else:
    for parameters in grid:
        kernel, k, model = parameters
        # KP-GNN
        script = f"python3 train_CSL.py --model_name={model} --kernel={kernel} --K={k}  --num_layer=2 --reprocess --sample --se_type={se_type}"
        os.system(script)
        # K-hop GNN
        # script = f"python train_CSL.py --kernel={kernel} --K={k} --num_layer=2 --wo_peripheral_edge --wo_peripheral_configuration --wo_path_encoding --reprocess"
        # os.system(script)