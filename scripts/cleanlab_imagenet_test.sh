#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=20G

# activate conda env
source activate $1

# srun python3 src/test_cleanlab.py \
python3 src/test_cleanlab.py \
    --multirun \
    callbacks=default_test_cleanlab \
    +callbacks@callbacks.posterioragreement=pametric \
    +callbacks/components@callbacks.posterioragreement.dataset=pa_cleanlab_valval \
    experiment=cleanlab/imagenet \
    checkpoint_metric=acc \
    model.net.net=resnet18 \
    model.net.pretrained=false \
    data.corrected_mislabelled=false \
    seed=0 \
    trainer=cpu \
    trainer.deterministic=true \
    +trainer.fast_dev_run=true \
    +name_logger=train_palabelckpt \
    logger.wandb.name=false_acc_18 \