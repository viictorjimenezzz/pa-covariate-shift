#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=10G
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4

srun python ./proves.py