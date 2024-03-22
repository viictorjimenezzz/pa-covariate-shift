#!/bin/bash

python3 src/train_dg_pa.py \
    --multirun \
    experiment=dg/pa_training \
    name=erm_trainerdebug \
    data.dg.envs_index=[0,1] \
    data.dg.envs_name=rebuttal \
    data.dg.disjoint_envs=True \
    data.dg.train_val_sequential=True \
    trainer.pa_datamodule.envs_index=[0,4] \
    trainer.pa_datamodule.shift_ratio=0.6 \
    trainer.pa_datamodule.envs_name=test_rebuttal \
    trainer=cpu \
    +trainer.limit_train_batches=0.015 \
    +trainer.limit_val_batches=0.07 \
    trainer.min_epochs=5 \
    trainer.max_epochs=5 \
    logger=none \
    # hydra/launcher=submitit_slurm \
    # hydra.launcher.tasks_per_node=1 \
    # hydra.launcher.mem_per_cpu=100000 \
    # +hydra.launcher.time=4:00:00 \
    # +hydra.launcher.num_gpus=4 \
    # logger.wandb.entity=malvai \
    # logger.wandb.project=cov_pa \
    # logger.wandb.group=pa_debugnew