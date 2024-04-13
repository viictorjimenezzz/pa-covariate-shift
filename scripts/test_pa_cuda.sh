#!/bin/bash

#SBATCH --mem-per-cpu=50G
#SBATCH --gpus=4

python tests/test_pa.py\
    # pa_module.trainer.ddp.devices=4 \
    # ddp.trainer.ddp.devices=4 \
    
    # must match with the number of gpus