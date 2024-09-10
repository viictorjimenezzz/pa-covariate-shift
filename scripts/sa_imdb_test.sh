#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=20G
#SBATCH --time=4:00:00

# python3 src/test_nlp.py \
srun python3 src/test_nlp.py \
    --multirun \
    experiment=sentiment_analysis_test \
    callbacks.posterioragreement.pa_epochs=1000 \
    callbacks.posterioragreement.dataset.perturbation=adversarial \
    callbacks.posterioragreement.dataset.intensity=4 \
    seed=0 \
    trainer=gpu \
    trainer.deterministic=true \
    experiment_name=imdb \
    checkpoint_metric=acc \

    # +callbacks@callbacks.posterioragreement=pametric \
    # +callbacks/components@callbacks.posterioragreement.dataset=pa_imdb \
    # 2,8,16,32,64,128,256,512,1024