#!/bin/bash

python3 src/train.py \
    --multirun \
    callbacks=default_train_modelselection_wilds \
    +callbacks@callbacks.posterioragreement=pametric_toremove \
    callbacks.posterioragreement.pairing_strategy=label_nn \
    +callbacks/components@callbacks.posterioragreement.dataset=pa_wilds_trainval \
    experiment=dg/wilds/waterbirds_erm \
    +data/dg/wilds@data=waterbirds \
    +auxiliary_args.dataconfname=waterbirds \
    name_logger=debug \
    data.transform.is_training=true \
    seed=0 \
    trainer=cpu \
    trainer.max_epochs=1 \
    trainer.deterministic=false \
    +trainer.fast_dev_run=true \
    trainer.limit_train_batches=0.0003029 \
    trainer.limit_val_batches=0.0000001 \
    callbacks.posterioragreement.cuda_devices=0 \
    callbacks.posterioragreement.num_workers=0 \
    logger.wandb=null \
    # +callbacks@callbacks.batch_size_finder=batch_size_finder_lisa \
    # logger.wandb.group=camelyon17 \
    # logger.wandb.project=cov_pa \
