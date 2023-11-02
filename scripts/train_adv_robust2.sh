#!/bin/bash

python src/train_pa.py \
    --multirun \
    experiment=adv/optimize_beta \
    model/adv/classifier@data.classifier=robust2 \
    data/adv/attack@data.attack=PGD \
    data.attack.epsilons=0.0314 \
    data.adversarial_ratio=0.1,0.5,0.9 \
    data.attack.steps=1000 \
    data.batch_size=1000 \
    trainer=ddp \
    +trainer.fast_dev_run=True \
    +trainer.limit_train_batches=1.0 \
    +trainer.limit_val_batches=1.0 \
    logger=wandb \
    hydra/launcher=submitit_slurm \
    hydra.launcher.tasks_per_node=1 \
    hydra.launcher.mem_per_cpu=10000 \
    +hydra.launcher.time=24:00:00 \
    +hydra.launcher.num_gpus=4 \
    logger.wandb.entity=malvai \
    logger.wandb.project=cov_pa \
    logger.wandb.group=task1_test
    # --cfg job \