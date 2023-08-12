#!/bin/bash
set -euo pipefail


CUDA_VISIBLE_DEVICES=2,3,4,5 python src/train_dg.py \
    experiment=dg/train_dg_erm \
    trainer.min_epochs=100 \
    trainer.max_epochs=100 \
    name=diagvib_weak \
    logger=wandb