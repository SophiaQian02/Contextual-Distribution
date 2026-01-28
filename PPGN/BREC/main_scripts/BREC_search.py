import os
SEED = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
for seed in SEED:
    script_base = f'python main_scripts/main_BREC.py --config configs/brec_config.json --LEARNING_RATE=1e-4 --WEIGHT_DECAY=1e-4 --BATCH_SIZE=32 --SEED={seed} --USE_KERNEL=True'
    print(script_base)
    os.system(script_base)