#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=20G
#SBATCH --time=24:00:00

# activate conda env
source activate $1

# python3 src/train.py \
srun python3 src/train.py \
    --multirun \
    callbacks=default_train_cleanlab \
    experiment=cleanlab/imagenet \
    seed=0 \
    trainer=gpu \
    trainer.max_epochs=1 \
    trainer.deterministic=true \
    +trainer.fast_dev_run=false \
    logger.wandb.name=debug \
    # +callbacks@callbacks.batch_size_finder=batch_size_finder_lisa \
    # +callbacks@callbacks.posterioragreement=pametric_label \
    # +callbacks/components@callbacks.posterioragreement.dataset=pa_wilds_trainval \
    # callbacks.posterioragreement.cuda_devices=0 \
    # callbacks.posterioragreement.num_workers=0 \