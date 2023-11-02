#!/bin/bash

python3 src/train_dg.py \
    --multirun \
    data.dg.env1_name=env1 \
    data.dg.env2_name=env2 \
    experiment=dg/train_dg_lisa \
    trainer=ddp \
    +trainer.fast_dev_run=False \
    +trainer.limit_train_batches=1.0 \
    +trainer.limit_val_batches=1.0 \
    trainer.min_epochs=100 \
    trainer.max_epochs=100 \
    name=nan \
    logger=none \
    hydra/launcher=submitit_slurm \
    hydra.launcher.tasks_per_node=1 \
    hydra.launcher.mem_per_cpu=50000 \
    +hydra.launcher.time=4:00:00 \
    +hydra.launcher.num_gpus=4
    #logger.wandb.entity=malvai \
    #logger.wandb.project=cov_pa \
    #logger.wandb.group=ZGO_position
    #name=lisa_ndisj_ZGO_position \