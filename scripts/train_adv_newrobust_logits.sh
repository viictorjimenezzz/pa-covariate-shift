#!/bin/bash

python src/train_pa.py \
    --multirun \
    experiment=adv/optimize_beta_logits \
    model/adv/classifier@data.adv.classifier=bpda \
    data/adv/attack=FMN \
    data.adv.adversarial_ratio=0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
    data.adv.attack.steps=1000 \
    data.adv.batch_size=1000 \
    trainer=ddp \
    logger=wandb \
    logger.wandb.entity=malvai \
    logger.wandb.project=cov_pa \
    logger.wandb.group=adversarial_pa \
    hydra/launcher=submitit_slurm \
    hydra.launcher.tasks_per_node=1 \
    hydra.launcher.mem_per_cpu=50000 \
    +hydra.launcher.time=4:00:00 \
    +hydra.launcher.num_gpus=4
    # data/adv/attack=PGD,GAUSSIAN,FMN \
    # model/adv/classifier@data.adv.classifier=weak,wong2020,addepalli2021,robust,wang2023,peng2023,bpda \
    # data.adv.adversarial_ratio=0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
    # data.adv.attack.epsilons=0.0314,0.0627,0.1255
    # data.adv.attack.steps=1000 \