#!/bin/bash
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --mem-per-cpu=50G
#SBATCH --time=120:00:00

# activate conda env
source activate $1

# python3 src/train.py \
srun python3 src/train.py \
    --multirun \
    callbacks=default_train_modelselection_wilds \
    +callbacks@callbacks.posterioragreement=pametric_toremove \
    callbacks.posterioragreement.pa_epochs=1000 \
    callbacks.posterioragreement.pairing_strategy=label_nn \
    +callbacks/components@callbacks.posterioragreement.dataset=pa_wilds_trainval \
    experiment=dg/wilds/fmow_erm \
    name_logger=erm_labnn \
    +data/dg/wilds@data=fmow_idtest \
    +auxiliary_args.dataconfname=fmow_idtest \
    data.transform.is_training=true \
    seed=0 \
    trainer=ddp \
    trainer.deterministic=true \
    # trainer.limit_train_batches=0.00001 \
    # trainer.limit_val_batches=0.02632 \
    # callbacks.model_checkpoint_PA.patience=2 \
    # +callbacks@callbacks.batch_size_finder=batch_size_finder_lisa \
    # callbacks.posterioragreement.pa_epochs=1000 \

    

    