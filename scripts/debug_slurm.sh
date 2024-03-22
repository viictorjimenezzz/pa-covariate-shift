#!/bin/bash
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --mem-per-cpu=10G

# activate conda env
source activate $1

srun python3 src/train_dg.py \
    --run \
    experiment=dg/train_dg_irm \
    test=true \
    +data/dg/wilds@data.dg=camelyon17_oracle \
    seed=123 \
    model.dg.net.net=densenet121 \
    data.dg.batch_size=16 \
    data.dg.num_workers=2 \
    data.dg.pin_memory=true \
    name=wilds_test_irm_oracle_full \
    trainer=ddp \
    +trainer.accumulate_grad_batches=2 \
    +trainer.replace_sampler_ddp=True \
    +trainer.fast_dev_run=False \
    trainer.min_epochs=1 \
    trainer.max_epochs=1 \
    logger=wandb \
    logger.wandb.entity=malvai \
    logger.wandb.project=cov_pa \
    logger.wandb.group=test_wilds \