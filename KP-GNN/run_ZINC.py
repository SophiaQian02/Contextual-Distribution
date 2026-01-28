import os
from itertools import product
import argparse

parser = argparse.ArgumentParser(f'add multi-gpu option')
parser.add_argument("--parallel", action="store_true",
                    help="If true, use DataParallel for multi-gpu training")
parallel_arg = parser.parse_args()
parallel = parallel_arg.parallel

datasets = ["ZINC"]
Ks = [16]
models = ["KEGINPrime"]
grid = product(datasets, models, Ks)

if parallel:
    for parameters in grid:
        dataset, model, K = parameters
        script = f"python train_ZINC.py --dataset_name={dataset} --model_name={model} --K={K} --num_layer={K+1} --residual  --parallel  --search --use_edge_kernel"
        os.system(script)
else:
    for parameters in grid:
        dataset, model, K = parameters
        script = f"python train_ZINC.py --dataset_name={dataset} --model_name={model} --K={K} --num_layer={K+1} --residual --search --use_edge_kernel"
        os.system(script)
