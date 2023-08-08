#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

python src/train_pa.py \
    # --cfg job
    experiment=adv/optimize_beta \
    data/adv/model@data.model=weak,robust \
    data/adv/attack@data.attack=PGD \
    data.attack.epsilons=0.0314,0.0627,0.1255 \
    data.attack.adversarial_ratio=0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
    # logger=wandb
