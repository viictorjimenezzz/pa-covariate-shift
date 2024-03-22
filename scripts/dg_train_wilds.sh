#!/bin/bash
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --mem-per-cpu=10G
#SBATCH --time=4:00:00

# activate conda env
source activate $1

# python3 src/train.py \
srun python3 src/train.py \
    --multirun \
    experiment=dg/wilds/camelyon17_erm \
    +data/dg/wilds@data=camelyon17_oracle \
    data.transform.is_training=true \
    seed=123 \
    name_logger=prova_configs_train_newconf_call \
    trainer=ddp \
    trainer.max_epochs=3 \
    +trainer.fast_dev_run=false \
    logger=wandb \
    logger.wandb.group=test_wilds \
    # +trainer.deterministic=true \