<div align="center">

# Rethinking Robustness in Machine Learning: A Posterior Agreement Approach

[![python](https://img.shields.io/badge/-Python3.9.9-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_1.10.0-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_1.9.1-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3.1-89b8cd)](https://hydra.cc/)
<!-- [![tests](https://github.com/ashleve/lightning-hydra-template/actions/workflows/test.yml/badge.svg)](https://github.com/ashleve/lightning-hydra-template/actions/workflows/test.yml) -->
<!-- [![code-quality](https://github.com/ashleve/lightning-hydra-template/actions/workflows/code-quality-main.yaml/badge.svg)](https://github.com/ashleve/lightning-hydra-template/actions/workflows/code-quality-main.yaml) -->
<!-- [![codecov](https://codecov.io/gh/ashleve/lightning-hydra-template/branch/main/graph/badge.svg)](https://codecov.io/gh/ashleve/lightning-hydra-template) <br> -->
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/ashleve/lightning-hydra-template#license)
[![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/ashleve/lightning-hydra-template/pulls)
<!-- [![contributors](https://img.shields.io/github/contributors/aretor/adv_pa_new.svg)](https://github.com/aretor/adv_pa_new/graphs/contributors) -->

<!--
Code for replicating the experiments in the paper: *"J. B. Carvalho,
A. Torcinovich, A. E. Cinà, L. Schönherr, J. M. Buhmann, Posterior Agreement
for Model Robustness Assessment in Covariate Shift Scenarios"*.
-->

Made with the [<kbd>lightning-hydra-template</kbd>](https://github.com/ashleve/lightning-hydra-template)


</div>

<br>

## Table of Contents
- [Setup](#setup)
- [Algorithm Selection](#algorithm-selection)
  - [Adversarial Setting](#adversarial-setting)
  - [Out-of-distribution setting](#out-of-distribution-setting)
- [Model Selection](#model-selection)
- [Algorithm Selection](#algorithm-selection)
- [Other Experiments](#other-experiments)
- [Additional Parameters](#additional-parameters)


## Setup

- Before running the scripts you need to create a `data`, `logs`, `outputs` and `results` folder. The repository structure should be as follows: 

```
pa-covariate-shift/
├── configs/                # Containing YAML configuration files
├── data/                   # Containing datasets and model checkpoints
├── logs/                   # Containing the logs of each execution
├── outputs/                # Containing the command output of each execution
├── results/                # Containing figures and tables resulting from the experiments
├── scripts/                # Containing pre-built scripts
├── src/                    # Containing the source code
|   ├── callbacks/          # Containing Callback implementations
|   ├── data/               # Containing LightningDataModule implementations
|   ├── models/             # Containing LightningModule implementations
|   ├── plot/               # Containing python scripts to generate plots
|   └── utils/              # Containing useful methods
├── tests/                  # Containing pre-built sanity checks
├── requirements.txt        # Containing python dependencies
└── README.md               # This file
```

- Create a `.env` file to define the environment variables.

```
PROJECT_ROOT="."
RES_DIR = # set to the desired location
DATA_DIR = "${RES_DIR}/data"
LOG_DIR = "${RES_DIR}/logs"
OUTPUT_DIR = "${RES_DIR}/outputs"
```

- Create the three data folders according to the `RES_DIR` variable you have set up, or run the `./scripts/create_soft_links.sh` to directly create soft links to the specified directory paths (useful if you want to store the data in remote
locations).

## Algorithm selection

We compared the robustness assessment capabilities of PA and accuracy-based metrics in the covariate shift setting. Adversarially-attacked and domain-shifted samples were generated under different conditions, namely varying the nature of the shift, its magnitude, and the proportion of affected samples.

### Adversarial setting

To create the adversarial datasets you can run
```bash
./scripts/generate_adv_data.sh <option>=<value>
```

To replicate the experiments you can run
```bash
./scripts/adv_eval.sh --<config>=<value>
```

The required parameters for both scripts are:
- `model/adv/classifier@model.net`: the attacked model (see options at `configs/model/adv/classifier/`) 
- `data/adv/attack@data.attack`: the attack (see options at `configs/data/adv/attack/`)
- `auxiliary_args.steps`: the attack number of steps (tested with `1000`)  
- `auxiliary_args.epsilons`: the attack power (only for PGD, tested with
`0.0314`, `0.0627` and `0.1255`)
- `data.batch_size`: the attack batch size (tested with `1000`)
- `data.adversarial_ratio`: the attack adversarial ratio, in $[0, 1]$ 
- `callbacks.posterioragreement.pa_epochs`: PA optimization epochs (tested with `500`).

### Out-of-distribution setting

To create the DiagVib-6 shifted datasets you can run
```bash
./scripts/generate_diagvib_data.sh
```

Make sure to uncomment the `datashift` configuration imports. To replicate the experiments you can train the models by running
```bash
./scripts/dg_diagvib_train_datashift.sh --<config>=<value>
```

The required parameters are:
- `experiment`: model to train (see options at `configs/experiment/dg/diagvibsix/`)
- `model.ppred`: adjusting the $p_{\text{sel}}$ parameter for LISA.

You can obtain the desired metrics on the shifted test environments by running
```bash
./scripts/dg_diagvib_test_datashift.sh --<config>=<value>
```

The required parameters are:
- `experiment`: model to test (see options at `configs/experiment/dg/diagvibsix/`)
- `model.ppred`: adjusting the $p_{\text{sel}}$ parameter for LISA.
- `auxiliary_args.pa_datashift.shift_ratio`: $SR$ values (results reported for `0.2`-`1.0`)
- `data.envs_index_test`: pairs of test environments (results reported for `[0,1]`-`[0,5]`)

## Model selection

In-distribution model selection experiments with specific shortcut opportunity configurations were conducted with specific settings of the DiagVib-6 data. To generate the data, you can run

```bash
./scripts/generate_diagvib_data.sh
```

Make sure to uncomment the `model selection configuration imports.

## Other experiments

- To reproduce the synthetic experiment with Bernoulli samples (Figure 1), you can run
```bash
python ./src/artificial_pipeline.py
```

- To reproduce results on the IMDB classification experiment (Appendix C), you can finetune the classifier for the task with

```bash
./scripts/sa_imdb.sh
```

You can obtain the reported results by running
```bash
./scripts/sa_imdb_test.sh
```

The required parameters are:
- `callbacks.posterioragreement.dataset.perturbation`: type of perturbation (see options)
- `callbacks.posterioragreement.dataset.intensity`: attack power $W$ (results reported for `0`-`8`)

- To reproduce the robustness - feature alignment experiments (Appendix E) you can include the `+callbacks=pa_pca` configuration to the out-of-distribution experiments.

## Additional parameters
### Debugging parameters

- `trainer`: (optional) set it to `cpu` in order to disable GPU usage (for debugging)
- `logger`: (optional) set it to `None` to disable W&B logging

### Multiple experiments
You can run multiple experiments by defining more values for one parameter,
separated by a comma (e.g., `data.attack.epsilons=0.0314,0.0627,0.1255`) and by
adding the option `--multirun`