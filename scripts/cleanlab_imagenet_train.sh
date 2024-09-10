#!/bin/bash
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --mem-per-cpu=100G
#SBATCH --time=120:00:00

# activate conda env
source activate $1

# python3 src/train.py \
srun python3 src/train.py \
    --multirun \
    callbacks=default_train_cleanlab \
    +callbacks@callbacks.posterioragreement=pametric_debug \
    callbacks.posterioragreement.epochs_to_log_beta=[0,2,4,6,8,10,12,14,16,18,20] \
    callbacks.posterioragreement.pairing_strategy=label \
    +callbacks/components@callbacks.posterioragreement.dataset=pa_cleanlab_valval \
    experiment=cleanlab/imagenet \
    auxiliary_args.resume_training=true \
    model.net.net=resnet18 \
    model.net.pretrained=true \
    model.optimizer.lr=0.001 \
    seed=0 \
    trainer=ddp \
    trainer.max_epochs=100 \
    trainer.precision=16 \
    trainer.deterministic=false \
    data.num_workers=72 \
    +trainer.fast_dev_run=false \
    logger.wandb.name=train18_vvc \