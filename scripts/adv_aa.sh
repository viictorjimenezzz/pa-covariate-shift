#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=20G
#SBATCH --time=24:00:00
#SBATCH --output=/cluster/home/vjimenez/ood-ppci/adv/%j.out
#SBATCH --error=/cluster/home/vjimenez/ood-ppci/adv/%j.err

# srun python3 src/train.py \
python3 src/train.py \
    --multirun \
    experiment=diagvib \
    trainer=cpu \
    experiment=adv/adv_aa \
    logger=none