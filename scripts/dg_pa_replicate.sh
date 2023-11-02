#!/bin/bash

python3 src/train_dg_pa.py \
    --multirun \
    experiment=dg/erm_irm_lisa_pa \
    data.dg.env1_name=t0_ndisj \
    data.dg.env2_name=t0_ndisj,t1_ndisj,t2_ndisj,t3_ndisj,t4_ndisj,t5_ndisj \
    data.dg.shift_ratio=0.2,0.4,0.6,0.8,1.0 \
    model.dg.classifier.exp_name=erm_ndisj,irm_ndisj,lisa_ndisj \
    trainer=ddp \
    +trainer.fast_dev_run=False \
    +trainer.limit_train_batches=1.0 \
    +trainer.limit_val_batches=0 \
    logger=wandb \
    hydra/launcher=submitit_slurm \
    hydra.launcher.tasks_per_node=1 \
    hydra.launcher.mem_per_cpu=50000 \
    +hydra.launcher.time=4:00:00 \
    +hydra.launcher.num_gpus=4 \
    logger.wandb.entity=malvai \
    logger.wandb.project=cov_pa \
    logger.wandb.group=erm_irm_lisa_dg_replicate