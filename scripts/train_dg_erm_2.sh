#!/bin/bash

python src/train_dg.py \
    experiment=dg/train_dg_erm \
    test=true \
    +data/dg/wilds@data.dg=camelyon17_oracle \
    seed=123 \
    model.dg.net.net=densenet121 \
    data.dg.batch_size=16 \
    data.dg.num_workers=2 \
    data.dg.pin_memory=true \
    name=wilds_test_erm_oracle_conf \
    trainer=ddp \
    +trainer.accumulate_grad_batches=2 \
    +trainer.replace_sampler_ddp=True \
    +trainer.fast_dev_run=False \
    +trainer.limit_train_batches=0.001 \
    +trainer.limit_val_batches=0.01 \
    +trainer.limit_test_batches=0.01 \
    trainer.min_epochs=1 \
    trainer.max_epochs=1 \
    logger=wandb \
    logger.wandb.entity=malvai \
    logger.wandb.project=cov_pa \
    logger.wandb.group=test_wilds \
    hydra/launcher=submitit_slurm \
    hydra.launcher.tasks_per_node=4 \
    hydra.launcher.gpus_per_node=4 \
    hydra.launcher.mem_per_cpu=10G \
    +hydra.launcher.time=4:00:00 \
    logger.wandb.entity=malvai \
    logger.wandb.project=cov_pa \
    logger.wandb.group=test_wilds \
    