import pandas as pd
import numpy as np

def select_and_remove_random_elements(arr, N):
    if N > len(arr):
        raise ValueError("N cannot be larger than the array size")
        
    random_indices = np.random.choice(len(arr), size=N, replace=False)

    selected_elements = arr[random_indices]
    new_arr = np.delete(arr, random_indices)
    return selected_elements, new_arr


def balance_dataset(ds_name, delete_remaining=False):
    # Load the dataset
    PATH_LOAD = "data/dg/dg_datasets/dataset/" + ds_name + "/metada.csv"
    PATH_SAVE = "data/dg/dg_datasets/dataset/" + ds_name + "/metada_b.csv"

    df = pd.read_csv(PATH_LOAD)
    df_0 = df[df['task_labels'] == 4].copy()
    class_0 = np.sort(df_0['Unnamed: 0'].to_numpy())
    df_1 = df[df['task_labels'] == 9].copy()
    class_1 = np.sort(df_1['Unnamed: 0'].to_numpy())

    # force symmetry
    to_keep = min(len(class_0), len(class_1))
    if len(class_0 != len(class_1)):
        df_0 = df_0.iloc[:to_keep,:]
        class_0 = class_0[:to_keep]
        df_1 = df_1.iloc[:to_keep,:]
        class_1 = class_1[:to_keep]# - (len(class_1) - to_keep)

    # number of blocks
    batch_size = 64
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

    
    if not delete_remaining:
        # Here is to account for the left over
        if (i+1)%2 == 0:
            classes = [class_0, class_1]
        else:
            classes = [class_1, class_0]
        new_class_0[-left:] = np.random.permutation(classes[0])
        new_class_1[-left:] = np.random.permutation(classes[1])


    df = pd.concat([df_0.iloc[:len_vec,:], df_1.iloc[:len_vec,:]])
    df['permutation'] = np.concatenate([new_class_0[:len_vec], new_class_1[:len_vec]])

    del df['Unnamed: 0']
    #df['prova'] = np.concatenate([class_00, class_10])
    df.reset_index(drop=True, inplace=True)
    df.set_index = np.concatenate([class_00, class_10])
    df.to_csv(PATH_SAVE, index=True)


for ds_name in ["train_env1", "train_env2", "validation", "test0", "test1", "test2", "test3", "test4", "test5"]:
#for ds_name in ["train_env1"]:
    balance_dataset(ds_name, delete_remaining=False)