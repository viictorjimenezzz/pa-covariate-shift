#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=100G

# activate conda env
source activate $1

# python3 src/test_datashift.py \
srun python3 src/test_datashift.py \
    --multirun \
    +callbacks/components@callbacks.posterioragreement.dataset=pa_diagvib_paper_train \
    +auxiliary_args.pa_datashift.shift_ratio=1.0 \
    experiment=dg/diagvibsix/diagvibsix_erm \
    data.envs_index=[0,1],[2,3],[4,5] \
    data.envs_name=env \
    data.disjoint_envs=False \
    experiment_name=erm_oracle \
    checkpoint_metric=acc \
    seed=0 \
    trainer=gpu \
    trainer.max_epochs=1 \
    trainer.deterministic=true \
    +trainer.fast_dev_run=false \
    logger.wandb=null \
    callbacks=default \
    # logger.wandb.group=diagvibsix_paper \