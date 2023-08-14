#!/bin/bash
set -euo pipefail

device=2,3,4,5
process_size=3
exp_names=("robust")
process=0
counter=0

# CUDA_VISIBLE_DEVICES=${device} python src/train_dg_pa.py \
#     experiment=dg/optimize_beta \
#     data.dg.ds1_env=test0 \
#     data.dg.ds2_env=test0 \
#     model.dg.classifier.exp_name=diagvib_erm_weak \
#     data.dg.shift_ratio=1 
#     # trainer=ddp
#     # logger=wandb

for exp_name in "${exp_names[@]}"; do  # models
    for env in $(seq 0 5); do
        for shift_ratio in $(seq 0.1 0.1 1); do
            log_file="outputs/dg/dg_script_${counter}_${exp_name}.log"
            CUDA_VISIBLE_DEVICES=${device} python src/train_dg_pa.py \
                experiment=dg/optimize_beta \
                data.dg.ds1_env=test0 \
                data.dg.ds2_env=test${env} \
                trainer.devices=1 \
                model.dg.classifier.exp_name=diagvib_${exp_name} \
                data.dg.shift_ratio=${shift_ratio} \
                logger=wandb \
                logger.wandb.group=dg_pa_test > "$log_file" 2>&1 &
            
            counter=$((counter+1))
            process=$(((process+1)%$process_size))
            if [ $process -eq 0 ]
            then 
                wait
            fi
        done
    done
done