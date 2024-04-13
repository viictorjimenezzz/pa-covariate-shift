#!/bin/bash

#SBATCH --job-name=download_imagenet
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=batch
#SBATCH --ntasks=1

# Go to the target directory
cd /cluster/scratch/vjimenez

# Validation data
# wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
# cd ../ && mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
# wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
# find . -name "*.JPEG" | xargs -I {} convert {} -resize "256^>" {}

# Training data
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
# mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
# tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
# find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done