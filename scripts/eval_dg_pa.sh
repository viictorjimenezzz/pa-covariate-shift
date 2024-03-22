#!/bin/bash

python3 src/train_dg_pa.py \
    --multirun \
    experiment=dg/erm_irm_lisa_pa_logits \
    exp_name=erm_rebuttalTFFF \
    logger.wandb.group=dg_pa_diagvib_TFFF \
    data.dg.envs_index=[0,1],[0,2],[0,3],[0,4],[0,5] \
    data.dg.shift_ratio=0.2,0.4,0.6,0.8,1.0 \
    data.dg.envs_name=test_rebuttaltf \
    trainer=ddp \
    +trainer.fast_dev_run=False \
    +trainer.limit_train_batches=1.0\
    trainer.limit_val_batches=1.0 \
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
    # logger.wandb.group=dg_pa_diagvib_new
    # exp_name=erm_rebuttal,irm_rebuttal,lisa_rebuttalD,lisa_rebuttalL \
    # data.dg.envs_index=[0,1],[0,2],[0,3],[0,4],[0,5] \
    # data.dg.shift_ratio=0.2,0.4,0.6,0.8,1.0 \
    