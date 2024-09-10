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

def _balance(df: pd.DataFrame, batch_size: int, num_shapes: int, obs_per_shape: int):
    # First, we remove the samples from each block (i.e. shape) that do not fit in a batch.
    minibatch = batch_size // num_shapes # each batch will be composed of num_shapes*minibatch samples
    samples_to_remove_from_each = obs_per_shape % minibatch

    perm_delete = np.ones(obs_per_shape, dtype=int)
    if samples_to_remove_from_each > 0:
        perm_delete[-samples_to_remove_from_each:] = -1
    df["permutation"] = np.concatenate([perm_delete]*num_shapes)

    # Now we delete them:
    df = df[df['permutation'] != -1]
    obs_per_shape = obs_per_shape - samples_to_remove_from_each
    num_batches = obs_per_shape // minibatch
    df.reset_index(drop=True, inplace=True)

    # Now we implement the balanced permutation.
    # Indexes 0:batch_size should have as indexes [0:minibatch, obs_per_shape:obs_per_shape+minibatch, ...]
    perm_counter = 0
    for j in range(num_batches):
        for i in range(num_shapes):
            # print(range(j*minibatch + i*obs_per_shape, (j+1)*minibatch + i*obs_per_shape - 1))
            # print(range(perm_counter, perm_counter + minibatch - 1))
            df.loc[perm_counter: perm_counter + minibatch - 1, "permutation"] = np.arange(j*minibatch + i*obs_per_shape, (j+1)*minibatch + i*obs_per_shape)
            perm_counter += minibatch

    return df

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

    # create the dataset folder for the validation
    os.makedirs(cfg.val_dataset_dir, exist_ok=True)

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
            if conf.randperm:
                df["permutation"] = np.random.permutation(conf.size*num_shapes) # np.arange(conf.size*num_shapes)
            else:
                # Balance the dataset so that each batch has observations from each shape.
                df = _balance(
                    df,
                    batch_size = cfg.batch_size, 
                    num_shapes = len(conf.shape), 
                    obs_per_shape = conf.size
                )

            path = cfg[f"{stage}_dataset_dir"] + stage + "_" + cfg.filename + str(e) + ".csv"
            if os.path.exists(path) == False:
                df.to_csv(path, index=False)            
        
    # Finally, we also store the configuration that generated such dataset:
    dict_cfg = OmegaConf.to_container(cfg, resolve=True)
    with open(cfg.val_dataset_dir + "config.json", 'w') as json_file:
        json.dump(dict_cfg, json_file, indent=4)

if __name__ == "__main__":
    main()