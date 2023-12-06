<div align="center">

# Posterior Agreement for Model Robustness Assessment in Covariate Shift Scenarios

[![python](https://img.shields.io/badge/-Python3.9.9-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_1.10.0-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_1.9.1-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3.1-89b8cd)](https://hydra.cc/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/) <br>
<!-- [![tests](https://github.com/ashleve/lightning-hydra-template/actions/workflows/test.yml/badge.svg)](https://github.com/ashleve/lightning-hydra-template/actions/workflows/test.yml) -->
<!-- [![code-quality](https://github.com/ashleve/lightning-hydra-template/actions/workflows/code-quality-main.yaml/badge.svg)](https://github.com/ashleve/lightning-hydra-template/actions/workflows/code-quality-main.yaml) -->
<!-- [![codecov](https://codecov.io/gh/ashleve/lightning-hydra-template/branch/main/graph/badge.svg)](https://codecov.io/gh/ashleve/lightning-hydra-template) <br> -->
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/ashleve/lightning-hydra-template#license)
[![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/ashleve/lightning-hydra-template/pulls)
<!-- [![contributors](https://img.shields.io/github/contributors/aretor/adv_pa_new.svg)](https://github.com/aretor/adv_pa_new/graphs/contributors) -->

Code for replicating the experiments in the paper: *"J. B. Carvalho,
A. Torcinovich, A. E. Cinà, L. Schönherr, J. M. Buhmann, Posterior Agreement
for Model Robustness Assessment in Covariate Shift Scenarios"*.

Made with the [<kbd>lightning-hydra-template</kbd>](https://github.com/ashleve/lightning-hydra-template)


</div>

<br>

## Before you start
Before running the scripts you need to create a `data`, `logs` and `outputs` folder.

- Create a `.env` file to define the environment variables. Set up the 
following three directory paths
    - `DATA_DIR`: where the datasets and trained models will be stored
    - `LOGS_DIR`: where the logs of the training will be stored
    - `OUTPUTS_DIR`: where the command line outputs will be stored
- Create the three data folders according to the variables you have set up, or
run the `./scripts/create_soft_links.sh` to directly create soft links to the 
specified directory paths (useful if you want to store the data in remote
locations)

## Adversarial Learning
To create the adversarial datasets you can run
```bash
python src/generate_adv_data.py experiment=adv/generate_adv_data <option>=<value> ...
```

To replicate the adversarial experiments you can run
```bash
python src/train_pa.py experiment=adv/optimize_beta <option>=<value> ...
```

The additional parameters for both scripts are:
- `model/adv/classifier@data.classifier`: the attacked model (`weak`, or
`roubst`) 
- `data/adv/attack@data.attack`: the attack (`PGD`, `FMN`)
- `data.attack.steps`: the attack number of steps (tested with `1000`)  
- `data.attack.batch_size`: the attack batch size (tested with `1000`)
- `data.attack.epsilons`: the attack power (only for PGD, tested with
`0.0314`, `0.0627` and `0.1255`)
- `data.adversarial_ratio`: the attack adversarial ratio, in $[0, 1]$ 
- `trainer`: (optional) set it to `cpu` in order to disable GPU usage (for debugging)
- `logger`: (optional) set it to `None` to disable W&B logging


## Domain Generalization

To replicate the adversarial experiments you can run
```bash
python src/train_pa.py experiment=adv/optimize_beta <option>=<val> ...
```

The additional parameters are:

- `data.dg.ds1_env`: the first dataset to compare with. Set it to `test0`
- `data.dg.ds2_env`: the other environments to compare with (`test0`, `test1`, `test2`, `test3`, `test4`, `test5`)
- `data.dg.shift_ratio`: the shift ratio, in $[0, 1]$
- `model.dg.classifier.exp_name`: the DG model (`diagvib_weak`,
`diagvib_robust`)

## Debugging parameters
- `trainer`: (optional) set it to `cpu` in order to disable GPU usage
- `logger`: (optional) set it to `None` to disable the default (W&B) logging

## Multiple experiments
You can run multiple experiments by defining more values for one parameter,
separated by a comma (e.g., `data.attack.epsilons=0.0314,0.0627,0.1255`) and by
adding the option `--multirun`
