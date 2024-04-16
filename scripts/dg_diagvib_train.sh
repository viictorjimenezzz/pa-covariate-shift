#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=100G
#SBATCH --time=24:00:00

# activate conda env
source activate $1

# python3 src/train.py \
srun python3 src/train.py \
    --multirun \
    callbacks=none \
    +data/dg/diagvib/modelselection@diagvib_dataset=hue_idval_4,hue_oodval_4 \
    experiment=dg/diagvibsix/diagvibsix_erm \
    logger.wandb.name=to_download \
    trainer.max_epochs=1 \
    seed=0 \
    trainer=gpu \
    trainer.deterministic=true \
    +trainer.fast_dev_run=false \
    # callbacks=default_train_modelselection \
    # +callbacks@callbacks.posterioragreement=pametric \
    # +callbacks/components@callbacks.posterioragreement.dataset=pa_diagvib_val_modelselection \