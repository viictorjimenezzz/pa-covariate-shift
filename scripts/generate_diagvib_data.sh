#!/bin/bash

#SBATCH --time=1-5
#SBATCH --mem-per-cpu=100G

python src/generate_diagvib_data.py