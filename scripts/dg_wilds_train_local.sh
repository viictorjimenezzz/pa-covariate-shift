#!/bin/bash

python3 src/train.py \
    --cfg=job \
    callbacks=default_train_modelselection \
    +callbacks@callbacks.posterioragreement=pametric_label \
    +callbacks/components@callbacks.posterioragreement.dataset=pa_wilds_trainval \
    experiment=dg/wilds/camelyon17_lisa \
    +data/dg/wilds@data=camelyon17_oracle \
    +auxiliary_args.dataconfname=camelyon17_oracle \
    name_logger=bsf_debug \
    data.transform.is_training=true \
    seed=0 \
    trainer=cpu \
    trainer.max_epochs=1 \
    trainer.deterministic=false \
    +trainer.fast_dev_run=false \
    trainer.limit_train_batches=0.0003029 \
    trainer.limit_val_batches=0.0000001 \
    callbacks.posterioragreement.cuda_devices=0 \
    callbacks.posterioragreement.num_workers=0 \
    logger.wandb=null \
    +callbacks@callbacks.batch_size_finder=batch_size_finder_lisa \
    # logger.wandb.group=camelyon17 \
    # logger.wandb.project=cov_pa \
