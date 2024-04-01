#!/bin/bash

python3 src/train.py \
    --multirun \
    callbacks=default_train_modelselection \
    +callbacks@callbacks.posterioragreement=pametric_label \
    +callbacks/components@callbacks.posterioragreement.dataset=pa_wilds_trainval \
    experiment=dg/wilds/rxrx1_erm \
    +data/dg/wilds@data=rxrx1_idtest \
    +auxiliary_args.dataconfname=rxrx1_idtest \
    name_logger=rxrx1_proves \
    data.transform.is_training=true \
    seed=0 \
    trainer=cpu \
    trainer.max_epochs=1 \
    trainer.deterministic=false \
    +trainer.fast_dev_run=false \
    trainer.limit_train_batches=0.000122 \
    trainer.limit_val_batches=0.000459 \
    callbacks.posterioragreement.cuda_devices=0 \
    callbacks.posterioragreement.num_workers=0 \
    logger.wandb=null \
    # logger.wandb.group=camelyon17 \
    # logger.wandb.project=cov_pa \