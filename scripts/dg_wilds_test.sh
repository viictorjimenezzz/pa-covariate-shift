#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=10G

# activate conda env
source activate $1

# python3 src/test.py \
srun python3 src/test.py \
    --multirun \
    experiment=dg/wilds/camelyon17_irm \
    +data/dg/wilds@data=camelyon17_oracle \
    data.transform.is_training=false \
    name_logger=irm_oracle \
    checkpoint_metric=acc,logPA \
    seed=0 \
    trainer=gpu \
    +trainer.fast_dev_run=false \
    trainer.deterministic=true \
    logger=wandb \
    logger.wandb.group=camelyon17 \
    # trainer.limit_test_batches=0.000189 \