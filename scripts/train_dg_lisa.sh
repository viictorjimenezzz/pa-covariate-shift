#!/bin/bash

python src/train_dg.py \
    --multirun \
    experiment=dg/train_dg_lisa \
    test=true \
    +data/dg/wilds@data.dg=camelyon17_oodval \
    seed=123 \
    model.dg.net.net=densenet121 \
    data.dg.batch_size=16 \
    data.dg.num_workers=2 \
    data.dg.pin_memory=true \
    name=wilds_test_lisa_oodval \
    model.dg.ppred=1.0 \
    model.dg.mix_alpha=2.0 \
    trainer=ddp \
    +trainer.fast_dev_run=False \
    +trainer.limit_train_batches=1.0 \
    +trainer.limit_val_batches=1.0 \
    +trainer.limit_test_batches=1.0 \
    trainer.min_epochs=1 \
    trainer.max_epochs=1 \
    logger=wandb \
    hydra/launcher=submitit_slurm \
    hydra.launcher.tasks_per_node=1 \
    hydra.launcher.mem_per_cpu=10G \
    +hydra.launcher.time=4:00:00 \
    +hydra.launcher.num_gpus=4 \
    logger.wandb.entity=malvai \
    logger.wandb.project=cov_pa \
    logger.wandb.group=test_wilds \
    # +hydra.launcher.additional_parameters.gpus=gtx_1080_ti:8 \
