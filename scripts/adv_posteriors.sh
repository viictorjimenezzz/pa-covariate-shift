#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=30G
#SBATCH --time=4:00:00

# python3 src/train_remove.py \
srun python3 src/train_remove.py \
    --multirun \
    callbacks=none \
    experiment=adv/eval_adv \
    +model/adv/classifier@model.net=weak,wong2020,wang2023,robust,addepalli2021,bpda \
    +data/adv/attack@data.attack=FMN \
    auxiliary_args.steps=1000 \
    data.adversarial_ratio=1.0 \
    seed=0 \
    trainer=gpu \
    trainer.deterministic=true \
    trainer.min_epochs=1 \
    logger.wandb=null \

    # weak,wong2020,wang2023,robust,addepalli2021,bpda
    # PGD,GAUSSIAN,FMN \
    # auxiliary_args.steps=1000 \
    # data.adversarial_ratio=0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
    # auxiliary_args.epsilons=0.0314,0.0627,0.1255
    # auxiliary_args.steps=1000