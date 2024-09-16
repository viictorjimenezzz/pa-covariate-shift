#!/bin/bash
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --mem-per-cpu=50G
#SBATCH --time=24:00:00

# activate conda env
source activate $1

# python3 src/train.py \
srun python3 src/train.py \
    --multirun \
    callbacks=default_train_modelselection \
    +callbacks@callbacks.posterioragreement=pametric \
    callbacks.posterioragreement.pa_epochs=1000 \
    +callbacks/components@callbacks.posterioragreement.dataset=pa_diagvib_val_modelselection \
    +data/dg/diagvib/modelselection@diagvib_dataset=pos_oodval_npair \
    auxiliary_args.diagvib_task=modelselection \
    experiment=dg/diagvibsix/diagvibsix_lisa \
    model.ppred=0.4 \
    model.optimizer._target_=torch.optim.Adam \
    model.optimizer.lr=0.0005 \
    seed=123 \
    trainer=ddp \
    trainer.deterministic=false \
    trainer.max_epochs=100 \

    # model.optimizer.lr=0.0001,0.0005,0.001,0.005,0.01 \
    # hue_zero,hue_idval,hue_mixval,hue_maxmixval,hue_oodval
    # pos_zero,pos_idval,pos_mixval,pos_maxmixval, pos_oodval

    # hue_zero_npair,hue_idval_npair,hue_mixval_npair,hue_maxmixval_npair,hue_oodval_npair
    # pos_zero_npair,pos_idval_npair,pos_mixval_npair,pos_maxmixval_npair,pos_oodval_npair
    