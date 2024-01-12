#!/bin/bash

python3 src/train_dg_pa.py \
    --multirun \
    experiment=dg/erm_irm_lisa_pa_logits \
    exp_name=lisa_rebuttalD \
    data.dg.envs_index=[0,4] \
    data.dg.shift_ratio=0.6 \
    data.dg.envs_name=test_rebuttal \
    data.dg.disjoint_envs=False \
    data.dg.train_val_sequential=False \
    trainer=ddp \
    +trainer.fast_dev_run=False \
    +trainer.limit_train_batches=1.0\
    trainer.limit_val_batches=1.0 \
    trainer.min_epochs=200 \
    trainer.max_epochs=200 \
    logger=wandb \
    hydra/launcher=submitit_slurm \
    hydra.launcher.tasks_per_node=1 \
    hydra.launcher.mem_per_cpu=100000 \
    +hydra.launcher.time=4:00:00 \
    +hydra.launcher.num_gpus=4 \
    logger.wandb.entity=malvai \
    logger.wandb.project=cov_pa \
    logger.wandb.group=pa_debugnew
    