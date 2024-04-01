#!/bin/bash
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --mem-per-cpu=100G
#SBATCH --time=8:00:00

# activate conda env
source activate $1

srun python3 src/train.py \
    --multirun \
    callbacks=default_train_datashift \
    +callbacks/components@callbacks.posterioragreement.dataset=pa_diagvib_paper_eval \
    experiment=dg/diagvibsix/diagvibsix_erm \
    name_logger=erm_paper \
    data.n_classes=2 \
    data.envs_index=[0,1]\
    data.envs_name=env \
    data.disjoint_envs=False \
    seed=0 \
    trainer=ddp \
    trainer.deterministic=true \
    +trainer.fast_dev_run=false \
    logger.wandb.group=paper_train \