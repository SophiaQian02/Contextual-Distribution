import os
SEED = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
for seed in SEED:
    script_base = f'python train_BREC.py --wo_path_encoding --se_type entropy --K=8 --kernel=spd --num_layer=8 --hidden_size=32 --BATCH_SIZE=32 --LEARNING_RATE=1e-4 --WEIGHT_DECAY=1e-4 --seed={seed} --model_name KPGIN'
    print(script_base)
    os.system(script_base)