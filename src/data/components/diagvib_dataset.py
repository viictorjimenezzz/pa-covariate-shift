import torch
import os
import pickle
from typing import Optional, List

from diagvibsix.data.dataset.dataset import Dataset, DatasetCSV
from diagvibsix.data.wrappers import TorchDatasetWrapper, get_per_ch_mean_std
from diagvibsix.data.auxiliaries import load_yaml

__all__ = ['DiagVib6Dataset', 'select_dataset_spec']


def select_dataset_spec(dataset_dir: str, dataset_name: str):
    """
    Given a dataset directory and a dataset name, provides with the corresponding specification file and/or cache file.
    Since this function works with the DiagvibDataModule, a cache_filepath will always be considered (not None).
    """
    yml_path = os.path.join(dataset_dir, dataset_name + ".yml")
    csv_path = os.path.join(dataset_dir, dataset_name + ".csv")
    cache_filepath = os.path.join(dataset_dir, dataset_name + ".pkl")
    
    if os.path.exists(yml_path):
        return yml_path, cache_filepath
    elif os.path.exists(csv_path):
        return csv_path, cache_filepath
    else:
        return None, cache_filepath


def check_conditions(dataset_specs_path, cache_path):
    # check the filepaths are correct
    if dataset_specs_path != None:
        is_csv = dataset_specs_path.endswith('.csv')
        is_yml = dataset_specs_path.endswith('.yml') or dataset_specs_path.endswith('.yaml')
        dataset_exists = os.path.exists(dataset_specs_path)
    else:
        is_csv = False
        is_yml = False
        dataset_exists = False

    if cache_path != None:
        is_pkl = cache_path.endswith('.pkl')
        cache_exists = os.path.exists(cache_path)
    else:
        is_pkl = False
        cache_exists = False
    
    
    # If only a CSV or YML file is provided
    if dataset_exists and (is_csv or is_yml) and not cache_exists:
        if is_pkl:
            print("Images will be generated from the CSV/YML file and stored in the cache file.")
        else:
            print("Images will be generated from the CSV/YML file and not stored.")
    # If only cache file is provided
    elif cache_exists and is_pkl and not dataset_exists:
        print("Dataset will be generated from the .pkl file.")
    # If both an existing cache file and cache path are provided
    elif dataset_exists and (is_csv or is_yml) and cache_exists and is_pkl:
        print("Dataset will be loaded from the cache file.")
    else:
        raise ValueError("Invalid or missing input files. Ensure you provide an existing CSV or YML file and/or a .pkl file.")
    
    return is_csv

# I STILL HAVE TO THINK ABOUT THE DATALOADER STRUCTURE.
class DiagVib6Dataset(TorchDatasetWrapper):
    def __init__(self,
                 mnist_preprocessed_path: str,
                 dataset_specs_path: Optional[str] = None,
                 cache_filepath: Optional[str] = None,
                 t: str = 'train',
                 seed: Optional[int] = 123,
                 normalization: Optional[str] = 'z-score', 
                 mean: Optional[List[float]] = None, 
                 std: Optional[List[float]] = None):
        
        # Check if the input files are valid
        is_csv = check_conditions(dataset_specs_path, cache_filepath)
        
        # Load dataset object (uint8 images) from cache if available
        if (cache_filepath is not None) and (os.path.exists(cache_filepath)):
            with open(cache_filepath, 'rb') as f:
                self.dataset = pickle.load(f)
        else:
            if is_csv:
                self.dataset = DatasetCSV(
                    mnist_preprocessed_path=mnist_preprocessed_path,
                    csv_path=dataset_specs_path,
                    t=t,
                    seed=seed)
            else:
                self.dataset = Dataset(
                    dataset_spec=load_yaml(dataset_specs_path), 
                    mnist_preprocessed_path=mnist_preprocessed_path,
                    cache_path=None,
                    seed=seed)

            # Save dataset object to cache if cache_path is provided
            if cache_filepath is not None:
                with open(cache_filepath, 'wb') as f:
                    pickle.dump(self.dataset, f, pickle.HIGHEST_PROTOCOL)

        self.unique_targets = sorted(list(set(self.dataset.task_labels)))

        self.normalization = normalization
        self.mean, self.std = mean, std
        self.min = 0.
        self.max = 255.
        if self.normalization == 'z-score' and (self.mean is None or self.std is None):
            self.mean, self.std = get_per_ch_mean_std(self.dataset.images)

    def __getitem__(self, item):
        sample = self.dataset.getitem(item)
        image, target, tag = sample.values()
        image = self._normalize(self._to_T(image, torch.float))
        #return {'image': image, 'target': target, 'tag': tag}
        return [image, self.unique_targets.index(target[1])] # we assume the task is the shape
    

# DEMOSTRACIO DE COM FUNCIONA: PREGUNTAR JOAO (especialment mida + task)
"""
from diagvibsix.data.dataset.preprocess_mnist import get_processed_mnist

PROCMNIST_PATH = get_processed_mnist("data/dg/")

ds = DiagVib6Dataset(
    mnist_preprocessed_path=PROCMNIST_PATH,
    dataset_specs_path="victor_proves.yml",
    cache_filepath="victor_proves_yml.pkl"
)

import matplotlib.pyplot as plt

for i in range(4):
    image, target, tag = ds.__getitem__(i).values()
    title = 'YML_' + str(i) + "_" + str(target) + "_" + str(tag)
    plt.imshow(image.permute(1, 2, 0))
    plt.savefig(title)


ds = DiagVib6Dataset(
    mnist_preprocessed_path=PROCMNIST_PATH,
    dataset_specs_path="victor_proves.csv",
    cache_filepath="victor_proves_csv.pkl"
)

import matplotlib.pyplot as plt

for i in range(4):
    image, target, tag = ds.__getitem__(i).values()
    title = 'CSV_' + str(i) + "_" + str(target) + "_" + str(tag)
    plt.imshow(image.permute(1, 2, 0))
    plt.savefig(title)
"""