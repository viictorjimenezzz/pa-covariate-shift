#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=30G
#SBATCH --time=24:00:00

# python3 src/train.py \
srun python3 src/train.py \
    --multirun \
    experiment=sentiment_analysis \
    seed=0 \
    trainer=gpu \
    trainer.max_epochs=100 \
    trainer.deterministic=true \
    data.num_workers=0 \
    data.batch_size=16 \