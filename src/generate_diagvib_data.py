import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything

import os
import numpy as np
import pandas as pd
import json

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Add resolver to automatically generate the size of the environment.
from src.utils.omegaconf import register_resolvers
register_resolvers()

from src import utils

@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="generate_diagvib_data",
)
def main(cfg: DictConfig):
    # print configuration tree
    utils.extras(cfg) 

    # set seed for random permutations
    if cfg.get("seed"): 
        seed_everything(cfg.seed, workers=True)

    # create the dataset folder
    os.makedirs(cfg.datasets_dir, exist_ok=True)

    for stage in cfg.dataset_specifications.keys():
        conf = cfg.dataset_specifications[stage]
        
        num_shapes = len(cfg.shape)
        for e, env in enumerate(conf.envs):
            dfs = []
            for shape in conf.shape:
                dfs.append(
                    pd.DataFrame(
                        {   "task_labels": shape*np.ones(conf.size).astype(int),
                            "shape": shape*np.ones(conf.size).astype(int),
                            "hue": env[0]*np.ones(conf.size).astype(int),
                            "lightness": env[1]*np.ones(conf.size).astype(int),
                            "texture": env[2]*np.ones(conf.size).astype(int),
                            "position": env[3]*np.ones(conf.size).astype(int),
                            "scale": env[4]*np.ones(conf.size).astype(int),
                        }
                    )
                )
            
            # Concatenate datafraemes and save as a CSV.
            df = pd.concat(dfs, ignore_index=True)
            df["permutation"] = np.random.permutation(conf.size*num_shapes) if conf.randperm else np.arange(conf.size*num_shapes)
            path = cfg.datasets_dir + stage + "_" + cfg.filename + str(e) + ".csv"
            df.to_csv(path, index=False)
        
    # Finally, we also store the configuration that generated such dataset:
    dict_cfg = OmegaConf.to_container(cfg, resolve=True)
    with open(cfg.datasets_dir + "config.json", 'w') as json_file:
        json.dump(dict_cfg, json_file, indent=4)


if __name__ == "__main__":
    main()