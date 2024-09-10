#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=10G
#SBATCH --time=24:00:00

# activate conda env
source activate $1

# python3 src/train.py \
srun python3 src/train.py \
    --multirun \
    callbacks=default_train_modelselection \
    +callbacks@callbacks.posterioragreement=pametric \
    +callbacks/components@callbacks.posterioragreement.dataset=pa_diagvib_val_modelselection \
    callbacks.posterioragreement.pairing_csv=null \
    +data/dg/diagvib/modelselection@diagvib_dataset=hue_idval_2 \
    experiment=flooding/diagvibsix_erm \
    model.loss.flood_level=0.01,0.05,0.1,0.15,0.2 \
    seed=0 \
    trainer=gpu \
    trainer.deterministic=true \
    +trainer.fast_dev_run=false \