#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=50G
#SBATCH --time=4:00:00

# activate conda env
source activate $1

# python3 src/data_to_cache.py \
srun python3 src/data_to_cache.py \
    --multirun \
    callbacks=none \
    experiment=dg/diagvibsix/diagvibsix_erm \
    auxiliary_args.diagvib_task=datashift \
    +data/dg/diagvib/datashift@diagvib_dataset=ZGO_hue_3 \
    data.val_disjoint_envs=false\
    trainer=gpu \
    logger.wandb=null \
    seed=0 \

    # CGO_1_hue,CGO_2_hue,CGO_3_hue,ZGO_hue_3,ZSO_hue_3
    # CGO_1_pos,CGO_2_pos,CGO_3_pos,ZGO_pos_3,ZSO_pos_3

    # hue_zero_npair,hue_idval_npair,hue_mixval_npair,hue_maxmixval_npair,hue_oodval_npair
    # pos_zero_npair,pos_idval_npair,pos_mixval_npair,pos_maxmixval_npair,pos_oodval_npair