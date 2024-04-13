#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=20G

# activate conda env
source activate $1

# python3 src/test_datashift.py \
# data.envs_index_test = IMPORTANT, I CAN TEST MORE THAN ONE... LETS SEE...
srun python3 src/test_datashift.py \
    --multirun \
    experiment=dg/diagvibsix/diagvibsix_irm \
    experiment_name=erm \
    checkpoint_metric=acc \
    data.n_classes=2 \
    data.envs_index=[1],[2],[3],[4],[5]  \
    data.envs_name=env \
    data.disjoint_envs=False \
    seed=0 \
    trainer=gpu \
    trainer.deterministic=true \
    +trainer.fast_dev_run=false \
    logger.wandb.group=paper_train \
    # trainer.limit_test_batches=0.0032 \
    # +callbacks/components@callbacks.posterioragreement.dataset=pa_diagvib_paper_train \
    # callbacks.posterioragreement.cuda_devices=0 \
    # callbacks.posterioragreement.num_workers=0 \
    # +auxiliary_args.pa_datashift.shift_ratio=1.0 \