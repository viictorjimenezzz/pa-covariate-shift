#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=20G
#SBATCH --time=24:00:00

# activate conda env
source activate $1

# python3 src/train.py \
srun python3 src/train.py \
    --multirun \
    callbacks=default_train_modelselection \
    +callbacks@callbacks.posterioragreement=pametric_nn \
    +callbacks/components@callbacks.posterioragreement.dataset=pa_wilds_trainval \
    experiment=dg/wilds/camelyon17_erm \
    +data/dg/wilds@data=camelyon17_oodval \
    +auxiliary_args.dataconfname=camelyon17_oodval \
    name_logger=to_download \
    data.transform.is_training=true \
    seed=0 \
    trainer=cpu \
    trainer.deterministic=true \
    logger.wandb.group=camelyon17 \
    trainer.max_epochs=1\
    # callbacks.model_checkpoint_PA.patience=2 \
    # +callbacks@callbacks.batch_size_finder=batch_size_finder_lisa \

    

    