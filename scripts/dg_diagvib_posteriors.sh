#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=80G
#SBATCH --time=4:00:00

# activate conda env
source activate $1

# python3 src/test_datashift.py \
srun python3 src/test_datashift.py \
    --multirun \
    callbacks=posteriors \
    callbacks.posteriors.optimal_beta=2.303 \
    auxiliary_args.project_name="DiagVib-6 Paper" \
    auxiliary_args.diagvib_task=datashift \
    +data/dg/diagvib/datashift@diagvib_dataset=paper_nonpaired \
    experiment=dg/diagvibsix/diagvibsix_erm \
    experiment_name=erm_001 \
    checkpoint_metric=acc \
    data.envs_index_test=[0,0],[0,1],[0,2],[0,3],[0,4],[0,5] \
    seed=0 \
    trainer=gpu \
    +trainer.fast_dev_run=false \
    logger.wandb=null

    # data.envs_index_test=[0,0],[0,1],[0,2],[0,3],[0,4],[0,5] \

    # trainer.limit_test_batches=0.003195 \
    # data.num_workers=0 \
    # data.pin_memory=false \
    # callbacks.posterioragreement.cuda_devices=0 \
    # callbacks.posterioragreement.num_workers=0 \    
    # data.envs_index_test=[0,1],[0,2],[0,3],[0,4],[0,5] \


    # CGO_1_hue,CGO_2_hue,CGO_3_hue,ZGO_hue_3,ZSO_hue_3
    # CGO_1_pos,CGO_2_pos,CGO_3_pos,ZGO_pos_3,ZSO_pos_3