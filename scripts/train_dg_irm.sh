#!/bin/bash

python src/train_dg.py \
    --multirun \
    experiment=dg/train_dg_irm \
    name=irm_rebuttal \
    data.dg.envs_index=[0,1] \
    data.dg.envs_name=rebuttal \
    data.dg.disjoint_envs=True \
    data.dg.train_val_sequential=True \
    trainer=ddp \
    +trainer.fast_dev_run=False \
    +trainer.limit_train_batches=1.0 \
    +trainer.limit_val_batches=1.0 \
    trainer.min_epochs=100 \
    trainer.max_epochs=100 \
    logger=wandb \
    hydra/launcher=submitit_slurm \
    hydra.launcher.tasks_per_node=1 \
    hydra.launcher.mem_per_cpu=100000 \
    +hydra.launcher.time=4:00:00 \
    +hydra.launcher.num_gpus=4 \
    logger.wandb.entity=malvai \
    logger.wandb.project=cov_pa \
    logger.wandb.group=pa_meeting