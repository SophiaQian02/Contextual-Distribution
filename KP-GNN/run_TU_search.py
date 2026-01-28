import os
from itertools import product
import argparse

parser = argparse.ArgumentParser(f'add multi-gpu option')
parser.add_argument("--parallel", action="store_true",
                    help="If true, use DataParallel for multi-gpu training")
parallel_arg = parser.parse_args()
parallel = parallel_arg.parallel

datasets = ["DD", "COLLAB"]
#datasets = ["MUTAG"]
#models = ["KPGCN", "KPGIN", "KPGraphSAGE"]
models = ["KPGIN"]
grid = product(datasets, models)

if parallel:
    for parameters in grid:
        dataset, model = parameters
        script = f"python train_TU.py --dataset_name={dataset} --model_name={model} --search --parallel --reprocess --sample"
        os.system(script)
else:
    for parameters in grid:
        dataset, model = parameters
        script = f"python train_TU.py --dataset_name={dataset} --model_name={model}   --device 1 --sample  --kernel2 reproduce --reprocess"
        os.system(script)
