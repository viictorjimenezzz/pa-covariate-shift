#!/bin/bash
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --mem-per-cpu=10G
#SBATCH --time=24:00:00

# activate conda env
source activate $1

# python3 src/train.py \
srun python3 src/train.py \
    --multirun \
    callbacks=default_train_modelselection \
    +callbacks@callbacks.posterioragreement=pametric_label \
    +callbacks/components@callbacks.posterioragreement.dataset=pa_wilds_trainval \
    experiment=dg/wilds/camelyon17_irm \
    +data/dg/wilds@data=camelyon17_idval \
    +auxiliary_args.dataconfname=camelyon17_idval \
    name_logger=irm_idval_debug3 \
    data.transform.is_training=true \
    seed=0 \
    trainer=ddp \
    trainer.max_epochs=10 \
    trainer.deterministic=true \
    logger.wandb.group=camelyon17