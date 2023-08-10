#!/bin/bash
set -euo pipefail


CUDA_VISIBLE_DEVICES=2,3,4,5 python src/train_dg.py \
    experiment=dg/train_dg_erm
    # --cfg=job \
    #logger=wandb
