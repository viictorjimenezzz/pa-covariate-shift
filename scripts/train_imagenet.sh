#!/bin/bash
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --mem-per-cpu=20G
#SBATCH --time=120:00:00

# activate conda env
source activate $1

# python3 src/train.py \
srun python3 src/train.py \
    --multirun \
    callbacks=default_train_cleanlab \
    +callbacks@callbacks.posterioragreement=pametric \
    +callbacks/components@callbacks.posterioragreement.dataset=pa_cleanlab_trainval \
    experiment=cleanlab/imagenet \
    seed=0 \
    trainer=ddp \
    trainer.max_epochs=100 \
    trainer.deterministic=true \
    +trainer.fast_dev_run=false \
    logger.wandb.name=train_palabelckpt \