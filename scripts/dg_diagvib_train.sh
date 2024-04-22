#!/bin/bash
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --mem-per-cpu=10G
#SBATCH --time=4:00:00

# activate conda env
source activate $1

# python3 src/train.py \
srun python3 src/train.py \
    --multirun \
    callbacks=default_train_modelselection \
    +callbacks@callbacks.posterioragreement=pametric \
    +callbacks/components@callbacks.posterioragreement.dataset=pa_diagvib_val_modelselection \
    +data/dg/diagvib/modelselection@diagvib_dataset=hue_idval_2L,hue_mixval_2,hue_mixval_4 \
    experiment=dg/diagvibsix/diagvibsix_irm \
    logger.wandb.name=irm \
    seed=0 \
    trainer=ddp \
    trainer.deterministic=true \
    +trainer.fast_dev_run=false \
    