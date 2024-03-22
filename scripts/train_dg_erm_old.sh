#!/bin/bash

python src/train_dg.py \
    --multirun \
    experiment=dg/train_dg_erm_old \
    data.dg.envs_index=[0,1] \
    data.dg.envs_name=singlevar \
    name=diagvib_pa \
    trainer=ddp \
    +trainer.fast_dev_run=False \
    +trainer.limit_train_batches=1.0 \
    +trainer.limit_val_batches=1.0 \
    trainer.min_epochs=10 \
    trainer.max_epochs=10 \
    logger=wandb \
    logger.wandb.entity=malvai \
    logger.wandb.project=cov_pa \
    logger.wandb.group=test_pametric \
    +hydra.launcher.additional_parameters.gpus=gtx_1080_ti:8 \
    hydra/launcher=submitit_slurm \
    hydra.launcher.tasks_per_node=4 \
    hydra.launcher.mem_per_cpu=10G \
    +hydra.launcher.time=4:00:00 \
    +hydra.launcher.num_gpus=4 \