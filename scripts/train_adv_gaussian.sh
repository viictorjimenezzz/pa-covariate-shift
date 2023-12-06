#!/bin/bash

python src/train_pa.py \
    --multirun \
    experiment=adv/optimize_beta \
    model/adv/classifier@data.classifier=weak \
    data/adv/attack@data.attack=GAUSSIAN \
    data.attack.epsilons=0.0314,0.0627,0.1255 \
    data.adversarial_ratio=1.0 \
    data.batch_size=1000 \
    logger=wandb \
    trainer=ddp \
    +trainer.fast_dev_run=False \
    +trainer.limit_train_batches=1.0 \
    +trainer.limit_val_batches=1.0 \
    hydra/launcher=submitit_slurm \
    hydra.launcher.tasks_per_node=1 \
    hydra.launcher.mem_per_cpu=10000 \
    +hydra.launcher.time=4:00:00 \
    +hydra.launcher.num_gpus=4 \
    logger.wandb.entity=malvai \
    logger.wandb.project=cov_pa \
    logger.wandb.group=gaussian_vs_PGD #\
    
    # --cfg job \
    #data.attack.noise_std=0.01,0.025,0.05,0.075,0.1,0.2,0.3,0.4,0.5 \