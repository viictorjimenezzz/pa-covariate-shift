#!/bin/bash

python src/train_pa.py \
    --multirun \
    experiment=adv/optimize_beta_logits \
    model/adv/classifier@data.adv.classifier=weak \
    data/adv/attack=GAUSSIAN \
    data.adv.adversarial_ratio=1.0 \
    data.adv.attack.epsilons=1.0 \
    data.adv.batch_size=1000 \
    trainer=ddp \
    logger=wandb \
    logger.wandb.entity=malvai \
    logger.wandb.project=cov_pa \
    logger.wandb.group=foo \
    hydra/launcher=submitit_slurm \
    hydra.launcher.tasks_per_node=1 \
    hydra.launcher.mem_per_cpu=50000 \
    +hydra.launcher.time=4:00:00 \
    +hydra.launcher.num_gpus=4
    # data/adv/attack=PGD,GAUSSIAN,FMN \
    # model/adv/classifier@data.adv.classifier=weak,wong2020,addepalli2021,robust,peng2023,bpda \
    # data.adv.adversarial_ratio=0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
    # data.adv.attack.epsilons=0.0314,0.0627,0.1255
    # data.adv.attack.epsilons=0.0,0.01960784,0.03921569,0.05882353,0.07843137,0.09803922,0.11764706,0.1372549,0.15686275,0.17647059,0.19607843,0.21568627,0.23529412,0.25490196,0.2745098,0.29411765,0.31372549,0.33333333,0.35294118,0.37254902,0.39215686,0.41176471,0.43137255,0.45098039,0.47058824,0.49019608,0.50980392,0.52941176,0.54901961,0.56862745,0.58823529,0.60784314,0.62745098,0.64705882,0.66666667,0.68627451,0.70588235,0.7254902,0.74509804,0.76470588,0.78431373,0.80392157,0.82352941,0.84313725,0.8627451,0.88235294,0.90196078,0.92156863,0.94117647,0.96078431,0.98039216,1.0 \
    # data.adv.attack.steps=1000 \