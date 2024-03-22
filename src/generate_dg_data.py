import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything

import os
import numpy as np
import pandas as pd
import json

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src import utils

def select_and_remove_random_elements(arr, N):

    if N > len(arr):
        raise ValueError("N cannot be larger than the array size")
        
    random_indices = np.random.choice(len(arr), size=N, replace=False)

    selected_elements = arr[random_indices]
    new_arr = np.delete(arr, random_indices)
    return selected_elements, new_arr

def balance_dataset(load_path: str, save_path: str, batch_size: int, delete_remaining = False):

    df = pd.read_csv(load_path)
    df_0 = df[df['task_labels'] == 4].copy()
    class_0 = np.sort(df_0['permutation'].to_numpy())
    df_1 = df[df['task_labels'] == 9].copy()
    class_1 = np.sort(df_1['permutation'].to_numpy())
    
    # force symmetry
    to_keep = min(len(class_0), len(class_1))
    if len(class_0 != len(class_1)):
        df_0 = df_0.iloc[:to_keep,:]
        class_0 = class_0[:to_keep]
        df_1 = df_1.iloc[:to_keep,:]
        class_1 = class_1[:to_keep]# - (len(class_1) - to_keep)

    # number of blocks
    block_size = int(batch_size/2)
    num_full_blocks = to_keep // block_size
    left = int((to_keep - block_size*num_full_blocks))

    if delete_remaining:
        # to delete the additional
        not_left = int(block_size*num_full_blocks)
        class_0 = class_0[:not_left]
        class_1 = class_1[:not_left]# - left
        len_vec = not_left
    else:
        len_vec = to_keep

    class_00 = class_0.copy()
    class_10 = class_1.copy()

    new_class_0 = np.zeros(len_vec, dtype=int)
    new_class_1 = np.zeros(len_vec, dtype=int)
    for i in range(num_full_blocks):
        start_idx = i * block_size

        inds_0, class_0 = select_and_remove_random_elements(class_0, block_size)
        inds_1, class_1 = select_and_remove_random_elements(class_1, block_size)

        if i%2 == 0:
            new_class_0[start_idx:start_idx+block_size] = inds_0
            new_class_1[start_idx:start_idx+block_size] = inds_1
        else: # exchange indexes
            new_class_0[start_idx:start_idx+block_size] = inds_1
            new_class_1[start_idx:start_idx+block_size] = inds_0

    
    if not delete_remaining and left > 0:
        # Here is to account for the left over
        if (i+1)%2 == 0:
            classes = [class_0, class_1]
        else:
            classes = [class_1, class_0]

        new_class_0[-left:] = np.random.permutation(classes[0])
        new_class_1[-left:] = np.random.permutation(classes[1])

    df = pd.concat([df_0.iloc[:len_vec,:], df_1.iloc[:len_vec,:]])
    del df['permutation']
    df['permutation'] = np.concatenate([new_class_0[:len_vec], new_class_1[:len_vec]])

    df.reset_index(drop=True, inplace=True)
    df.set_index = np.concatenate([class_00, class_10])
    df.to_csv(save_path, index=False)


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="generate_dg_data",
)
def main(cfg: DictConfig):
    utils.extras(cfg) # print conf tree

    if cfg.get("seed"): # impose seed
        seed_everything(cfg.seed, workers=True)

    DATASETS_DIR = cfg.get("DATASETS_DIR")
    os.makedirs(DATASETS_DIR, exist_ok=True)

    TO_BALANCE = cfg.get("BALANCE")
    file_name = cfg.get("file_name")

    SIZE_TRAIN = cfg.get("SIZE_TRAIN")
    SIZE_VAL = cfg.get("SIZE_VAL")
    SIZE_TEST = cfg.get("SIZE_TEST")
    for sh in [SIZE_TRAIN, SIZE_VAL, SIZE_TEST]:
        if sh%2 != 0:
            raise ValueError("Size must be even.")

    ## TRAINING & VALIDATION ---------------------------------------------
    # hue, lightness, texture, position, scale
    train_val_especs = cfg.get("train_val_especs")
    train_val_envs = list(train_val_especs.keys()) # number of environments
    tran_val_randperm = cfg.get("train_val_randperm")
    
    sizes = [SIZE_TRAIN, SIZE_VAL]
    tasks = ["train", "val"]
    for tv in range(2): # train and val
        SIZE = sizes[tv]
        task = tasks[tv]

        four = np.ones(int(SIZE/2))*4
        nine = np.ones(int(SIZE/2))*9
        task_label = np.concatenate([four, nine]).astype(int)
        shape = task_label.copy()

        for t in range(len(train_val_envs)):
            config = train_val_especs[train_val_envs[t]]

            if tran_val_randperm:
                permutation = np.random.permutation(SIZE)
            else:
                permutation = np.arange(SIZE)

            df = pd.DataFrame(
                {
                    "task_labels": task_label,
                    "shape": shape,
                    "hue": config[0]*np.ones(SIZE).astype(int),
                    "lightness": config[1]*np.ones(SIZE).astype(int),
                    "texture": config[2]*np.ones(SIZE).astype(int),
                    "position": config[3]*np.ones(SIZE).astype(int),
                    "scale": config[4]*np.ones(SIZE).astype(int),
                    "permutation": permutation.astype(int)
                }
            )

            path = DATASETS_DIR + task + "_" + file_name + str(t) + ".csv"
            df.to_csv(path, index=False)
            if TO_BALANCE:
                balance_dataset(path, path, batch_size = cfg.get("BATCH_SIZE"))


    # TEST ---------------------------------------------
    four = np.ones(int(SIZE_TEST/2))*4
    nine = np.ones(int(SIZE_TEST/2))*9
    task_label = np.concatenate([four, nine]).astype(int)
    shape = task_label.copy()

    # hue, lightness, texture, position, scale
    test_especs = cfg.get("test_especs")
    test_envs = list(test_especs.keys())
    for t in range(len(test_envs)):
        config = test_especs[test_envs[t]]

        df = pd.DataFrame(
                {
                    "task_labels": task_label,
                    "shape": shape,
                    "hue": config[0]*np.ones(SIZE_TEST).astype(int),
                    "lightness": config[1]*np.ones(SIZE_TEST).astype(int),
                    "texture": config[2]*np.ones(SIZE_TEST).astype(int),
                    "position": config[3]*np.ones(SIZE_TEST).astype(int),
                    "scale": config[4]*np.ones(SIZE_TEST).astype(int),
                    "permutation": np.arange(SIZE_TEST).astype(int)
                }
            )

        path = DATASETS_DIR + "test_" + file_name + str(t) + ".csv"
        df.to_csv(path, index=False)
        if t == 0 and TO_BALANCE: # we balance the first, and then copy the permutation to the rest
            balance_dataset(path, path, batch_size = cfg.get("BATCH_SIZE"))
            df = pd.read_csv(path)
            master_permutation = df['permutation']

        if t > 0 and TO_BALANCE:
            df = pd.read_csv(path)
            df['permutation'] = master_permutation
            df.to_csv(path, index=False)

        # Finally, we also store the configuration that generated such dataset:
        dict_cfg = OmegaConf.to_container(cfg, resolve=True)
        with open(DATASETS_DIR + "config.json", 'w') as json_file:
            json.dump(dict_cfg, json_file, indent=4)


if __name__ == "__main__":
    main()