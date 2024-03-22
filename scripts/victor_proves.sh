#!/bin/bash

#SBATCH --mem-per-cpu=10G

python ./victor_proves.py \
    --multirun \
    hydra/launcher=submitit_slurm \
    hydra.launcher.tasks_per_node=1 \
    hydra.launcher.mem_per_cpu=50000 \
    +hydra.launcher.time=4:00:00