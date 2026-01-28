# CUDA_VISIBLE_DEVICES=1 python main_brec.py --cfg configs/GPS/brec-GPS+VonEntropy.yaml  wandb.use False
# CUDA_VISIBLE_DEVICES=3 python main_brec.py --cfg configs/GPS/brec-GPS+VonKernel.yaml  wandb.use False
# CUDA_VISIBLE_DEVICES=1 python main_brec.py --cfg configs/GPS/brec-GPS+RWSE.yaml  wandb.use False
CUDA_VISIBLE_DEVICES=3 python main_brec.py --cfg configs/GPS/brec-GPS+LapPE.yaml  wandb.use False