#!/bin/bash
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --mem-per-cpu=100G
#SBATCH --time=24:00:00

# activate conda env
source activate $1

# python3 src/train.py \
srun python3 src/train.py \
    --multirun \
    callbacks=default_train_modelselection \
    +callbacks@callbacks.posterioragreement=pametric \
    +callbacks/components@callbacks.posterioragreement.dataset=pa_diagvib_val_modelselection \
    +data/dg/diagvib/modelselection@diagvib_dataset=hue_idval_2,hue_idval_10,hue_oodval_2,hue_oodval_10 \
    experiment=dg/diagvibsix/diagvibsix_lisa \
    +logger.wandb.name=lisa_10_10ep \
    model.ppred=1.0\
    seed=0 \
    trainer=ddp \
    trainer.max_epochs=10 \
    trainer.deterministic=true \
    +trainer.fast_dev_run=false \