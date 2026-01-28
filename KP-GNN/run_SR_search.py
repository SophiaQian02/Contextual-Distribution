import os
from itertools import product
import argparse

parser = argparse.ArgumentParser(f'add multi-gpu option')
parser.add_argument('--se_type', type=str, default="entropy", choices=("entropy", "RWSE", "LapSE"),
                    help="The type of structural encoding to use")
args = parser.parse_args()
se_type = args.se_type

kernels = ["gd"]
ks = [4]
models = ["KPGIN"]
grid = product(kernels, ks, models)
for parameter in grid:
    kernel, k, model = parameter
    # KP-GNN
    script = f"python train_SR.py --model_name={model} --K={k} --num_layer=2 --kernel={kernel} --wo_path_encoding --se_type={se_type} "
    os.system(script)
    # K-hop GNN
    # script = f"python train_SR.py --K={k} --num_layer=2 --kernel={kernel} --wo_path_encoding --wo_peripheral_edge --wo_peripheral_configuration"
    # os.system(script)