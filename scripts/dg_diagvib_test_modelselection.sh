#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=20G

# activate conda env
source activate $1

# python3 src/test_modelselection.py \
srun python3 src/test_modelselection.py \
    --multirun \
    callbacks=default_test_modelselection \
    +data/dg/diagvib/modelselection@diagvib_dataset=hue_idval_2,hue_idval_10,hue_oodval_2,hue_oodval_10 \
    experiment=dg/diagvibsix/diagvibsix_erm \
    experiment_name=erm,erm_10ep \
    checkpoint_metric=acc,logPA,AFR_pred \
    data.envs_index_test=[0],[1],[2],[3],[4] \
    seed=0 \
    trainer=gpu \
    trainer.deterministic=true \
    +trainer.fast_dev_run=false \
    +auxiliary_args.new_id=true \
    # trainer.limit_test_batches=0.0032 \