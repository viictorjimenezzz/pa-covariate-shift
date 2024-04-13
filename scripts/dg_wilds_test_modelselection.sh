#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=10G

# activate conda env
source activate $1

# srun python3 src/test_modelselection.py \
python3 src/test_modelselection.py \
    --multirun \
    experiment=dg/wilds/camelyon17_lisa \
    +data/dg/wilds@data=camelyon17_oracle \
    data.transform.is_training=false \
    name_logger=lisa_oracle_bsf_10 \
    checkpoint_metric=acc,logPA,AFR_pred \
    seed=0 \
    trainer=gpu \
    +trainer.fast_dev_run=false \
    trainer.deterministic=true \
    logger.wandb.group=camelyon17 \
    # trainer.limit_test_batches=0.00000189 \
   
    