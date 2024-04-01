#!/bin/bash

python3 src/train.py \
    --multirun \
    callbacks=default_train_datashift \
    +callbacks/components@callbacks.posterioragreement.dataset=pa_diagvib_local \
    +auxiliary_args.pa_datashift.shift_ratio=0.5 \
    experiment=dg/diagvibsix/diagvibsix_erm \
    name_logger=dg_diagvib_erm_local \
    data.n_classes=2 \
    data.envs_index=[0,1]\
    data.envs_name=singlevar \
    data.disjoint_envs=False \
    seed=0 \
    trainer=cpu \
    trainer.deterministic=true \
    +trainer.fast_dev_run=false \
    trainer.max_epochs=5 \
    callbacks.posterioragreement.cuda_devices=0 \
    callbacks.posterioragreement.num_workers=0 \
    logger.wandb=null \
    # logger.wandb.group=diagvibsix_paper \
    # trainer.limit_train_batches=0.000000122 \
    # trainer.limit_val_batches=0.0000000459 \