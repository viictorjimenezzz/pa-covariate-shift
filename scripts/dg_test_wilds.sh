#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=10G

# activate conda env
source activate $1

# python3 src/test.py \
srun python3 src/train.py \
    --cfg=job \
    experiment=dg/camelyon17_erm \
    +data/dg/wilds@data=camelyon17_oracle \
    data.transform.is_training=false \
    seed=123 \
    name_logger=prova_configs_train_newconf \
    trainer=cpu \
    +trainer.fast_dev_run=true \
    logger=none \
    # logger.wandb.group=test_wilds \