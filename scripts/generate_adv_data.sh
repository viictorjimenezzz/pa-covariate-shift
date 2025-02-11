#!/bin/bash
set -euo pipefail

device=0
attack=FMN
# epsilons=("0.0314" "0.0627" "0.1255")
models=("weak" "robust")


if [ "$attack" = "FMN" ]; then
    for model in "${models[@]}"; do
        log_file="script_${device}.log"
        CUDA_VISIBLE_DEVICES=$device python src/generate_adv_data.py \
            experiment=adv/generate_adv_data \
            model/adv/classifier@model.net=$model \
            data/adv/attack@data.attack=$attack \
            data.attack.steps=1000 \
            data.batch_size=1000 \
            data.adversarial_ratio=1.0 > "$log_file" 2>&1 &

        device=$((device+1))
    done
else
    for model in "${models[@]}"; do
        for epsilon in "${epsilons[@]}"; do
            log_file="script_${device}.log"
            CUDA_VISIBLE_DEVICES=$device python src/generate_adv_data.py \
                experiment=adv/generate_adv_data \
                model/adv/classifier@model.net=$model \
                data/adv/attack@data.attack=$attack \
                data.attack.epsilons=$epsilon \
                data.attack.steps=1000 \
                data.batch_size=1000 \
                data.adversarial_ratio=1.0 > "$log_file" 2>&1 &

            device=$((device+1))
        done
    done
fi

wait