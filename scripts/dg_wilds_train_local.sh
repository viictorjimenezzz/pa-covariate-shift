#!/bin/bash
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --mem-per-cpu=10G
#SBATCH --time=4:00:00

# activate conda env
source activate $1

# srun python3 src/train.py \
python3 src/train.py \
    --multirun \
    experiment=dg/wilds/camelyon17_lisa \
    +data/dg/wilds@data=camelyon17_oracle \
    data.transform.is_training=true \
    name_logger=lisa_foo \
    seed=0 \
    trainer=cpu \
    trainer.max_epochs=1 \
    trainer.deterministic=false \
    +trainer.fast_dev_run=true \
    trainer.limit_train_batches=0.000122 \
    trainer.limit_val_batches=0.000459 \
    callbacks.posterioragreement.cuda_devices=0 \
    callbacks.posterioragreement.num_workers=0 \
    # logger=wandb \
    # logger.wandb.group=camelyon17 \
    # logger.wandb.project=cov_pa \