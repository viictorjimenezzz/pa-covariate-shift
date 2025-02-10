#!/bin/bash
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --mem-per-cpu=80G
#SBATCH --time=120:00:00

# activate conda env
source activate $1

# python3 src/train.py \
srun python3 src/train.py \
    --multirun \
    callbacks=default_train_datashift \
    auxiliary_args.diagvib_task=datashift \
    auxiliary_args.project_name="DiagVib-6 Paper" \
    +callbacks@callbacks.posterioragreement=pametric \
    +callbacks/components@callbacks.posterioragreement.dataset=pa_diagvib_datashift_val \
    callbacks.posterioragreement.pa_epochs=1000 \
    +data/dg/diagvib/datashift@diagvib_dataset=paper_nonpaired \
    experiment=dg/diagvibsix/diagvibsix_lisa \
    model.ppred=0.75 \
    model.optimizer._target_=torch.optim.Adam \
    model.optimizer.lr=0.001 \
    seed=0 \
    trainer=ddp \
    trainer.deterministic=true \
    +trainer.fast_dev_run=false \
    logger.wandb.name=lisa_075 \
    # data.batch_size=16 \

    # model.ppred=0.4 \
    # CGO_1_hue,CGO_2_hue,CGO_3_hue,ZGO_hue_3,ZSO_hue_3
    # CGO_1_pos,CGO_2_pos,CGO_3_pos,ZGO_pos_3,ZSO_pos_3
    # model.ppred=1.0 \
    
