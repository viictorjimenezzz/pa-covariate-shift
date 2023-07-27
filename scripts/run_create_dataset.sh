#!/bin/bash
# run this script from the root folder
set -euo pipefail

cd "$(dirname "$0")/.."

attack="PGD"
classifier="robust"
epsilons="0.0314"

# if [ $attack = "PGD" ]; then
#     epsilons=$(IFS=","; printf '%s' "0.0314 0.0627 0.1255")
# fi

cmd="python3 src/generate_adv_data.py \
experiment=adv/generate_adv_data \
data/adv/attack=${attack} \
model/adv/classifier=${classifier} \
data.adv.attack.params.epsilons=${epsilons};"

sbatch \
    -J adv_pa \
    -o outputs/generate_adv_data_att=${attack}_clf=${classifier}_eps=${epsilons} \
    --ntasks-per-node=1 \
    --time=120:00:00 \
    --mem-per-cpu=10000 \
    --gpus=1 \
    --wrap "$cmd"

# for i in $epsilons; do
#     exp_dir="exp_adv_${attack}_pr_20230512/exp_adv_${attack}_pr_esp${i}"
#     exp_name="exp_${attack}_model${model}_eps${i}_pr0"

#     exp_path="$exp_dir/$exp_name"
#     outputs_path="/cluster/project/jbuhmann/posterior_agreement/adv_pa/outputs/$exp_name"

#     # Run DG
#     # cmd="python3 evaluate_dg.py experiment=$exp_path;"
#     # # echo $cmd
#     # sbatch -J dg_pa -o "$outputs_path" --ntasks-per-node=1 --time=4:00:00 --mem-per-cpu=10000 --gpus=1 --wrap "$cmd"
#     # echo sbatch -J dg_pa -o "$outputs_path" --ntasks-per-node=1 --time=4:00:00 --mem-per-cpu=10000 --gpus=1 --wrap "$cmd"

#     # Run Adv
#     cmd="python3 evaluate_adv.py experiment=$exp_path;"
#     # echo $cmd
#     sbatch -J adv_pa -o "$outputs_path" --ntasks-per-node=1 --time=120:00:00 --mem-per-cpu=10000 --gpus=1 --wrap "$cmd"
#     echo sbatch -J adv_pa -o "$outputs_path" --ntasks-per-node=1 --time=120:00:00 --mem-per-cpu=10000 --gpus=1 --wrap "$cmd"
# done

