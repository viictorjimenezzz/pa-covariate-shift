import torch
import os
import pickle
from typing import Optional, List

from diagvibsix.data.wrappers import TorchDatasetWrapper, get_per_ch_mean_std
from diagvibsix.data.auxiliaries import load_yaml

from pandas import read_csv
import numpy as np
from numpy.random import seed as set_seed
from diagvibsix.data.dataset.dataset import Dataset, random_choice
from diagvibsix.data.dataset.dataset_utils import sample_attribute
from diagvibsix.data.dataset.paint_images import Painter
from diagvibsix.data.dataset.config import DATASETS, OBJECT_ATTRIBUTES

__all__ = ['DiagVib6DatasetPA', 'select_dataset_spec']

class SampleIterator:
    """
    Iterates indexes from `start` to `end`, and when `end` is reached, starts again from `start`.
    """
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.iterator = iter(range(start, end))

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            # Reset the iterator to start from the beginning of the range
            self.iterator = iter(range(self.start, self.end))
            return next(self.iterator)


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


class DatasetCSV_PA(Dataset):
    """Subclass of DiagVib dataset to generate images from customized CSV specifications for PA purposes. 
    In this case, training and validation images for each environment can be generated from disjoint datasets and set to be non-repetitive, and test images will be generated in the same sequence for all environments.
    
    Args:
        mnist_preprocessed_path (str): Path to the processed MNIST dataset. If there is no such dataset, you can generate it (e.g. in the DataLoader) by calling process_mnist.get_processed_mnist(mnist_dir).
        csv_path (str): Path to the CSV file containing the dataset specifications.
        t (str): Type of dataset to be generated, corresponding to the sample pool from where images will be generated. The possible values are 'train', 'val' and 'test'.
        split_numsplit (Optional[List[int]]): List containing the current environment index, and the total number of environments. This argument must be specified only if train/val environments are set to be disjoint. If no argument is passed, the generation of all environments will come from the same training or validation sample pool.
        train_val_sequential (bool): Boolean to indicate whether train/val datasets will be formed by taking a different sample each time. If the number of requested images for a class is smaller than the total number of images for that class, data in the same epoch will be non-repetitive. Test datasets are built this way by default.
        seed (Optional[int]): Random seed for the dataset generation.

    The CSV file must contain, in order:
        - `task_labels`: A column for the target associated with the task.
        - A column for each of the OBJECT_ATTRIBUTES.
        - `permutation`: A column for the permutation value. This is not used during the generation of the data, but determines the order in which generated samples are accessed.
    """

    def __init__(self,
                mnist_preprocessed_path: str,
                csv_path: str,
                t: str = 'train',
                split_numsplit: Optional[List[int]] = [0, 1],
                train_val_sequential: Optional[bool] = False,
                seed: Optional[int] = 123):
        
        set_seed(seed) # numpy bc thats how files are generated
        
        self.painter = Painter(mnist_preprocessed_path)
        self.metadata = read_csv(csv_path)
        self.permutation = self.metadata.permutation.to_list()
        self.task_labels = self.metadata.task_labels.to_list()

        self.len_factors = len(OBJECT_ATTRIBUTES.keys())

        # Needed to avoid overriding methods
        self.spec = {} 
        self.task = 'shape'
        self.spec['shape'] = [1, 128, 128] # MNIST expected shape

        # Computing lists of iterables
        self.current_split = split_numsplit[0]
        self.total_splits = split_numsplit[1]

        num_train_samples = [DATASETS['train']['samples'][i] for i in range(10)] 
        numels_split = [numsamp // self.total_splits for numsamp in num_train_samples]
        self.iterables_train = [SampleIterator(self.current_split*n, (self.current_split+1)*n) for n in numels_split]

        num_val_samples = [DATASETS['val']['samples'][i] for i in range(10)] 
        numels_split = [numsamp // self.total_splits for numsamp in num_val_samples]
        self.iterables_val = [SampleIterator(self.current_split*n, (self.current_split+1)*n) for n in numels_split]

        self.iterables_test = [SampleIterator(0,i) for i in DATASETS['test']['samples']]
        self.train_val_sequential = train_val_sequential

        # Generating the images
        self.images = []
        for index, row in self.metadata.iterrows():
            obj_spec = {col: OBJECT_ATTRIBUTES[col][val] for col, val in row.iloc[1:self.len_factors+1].items()}
            obj_spec['category'] = t

            mode_spec = {
                'tag': '',
                'objs': [obj_spec]
            }
            image_specs, images, env_label = self.draw_mode(mode_spec, 1) # 1 image per mode
            self.images += images

    
    def draw_image_spec_from_mode(self, mode_spec):
        """ Draws a single image specification from a mode.
        This function is modified from the original class to account for the iterable structure defined previously.
            
        Args:
            mode_spec (dict): The specification dictionary of the mode. It contains the keys `objs` and `tag`.

        Returns:
            image_spec (dict): A dictionary with keys 'tag' and 'obj', the latter one containing a list of the mode_spec['objs'] with the numeric values for each factor according to the selected category for each.
            semantic_image_spec (dict): Equivalent to image_spec, but with the names of the selected category for each factor.
        """
        # Set empty dictionary for each sample.
        image_spec = dict()

        # Add tag to image specification
        image_spec['tag'] = mode_spec['tag'] if 'tag' in mode_spec.keys() else ''

        # Each attribute in a mode can be given as a list (e.g. 'color': ['red', 'blue', 'green']).
        # In such cases we want to sample an attribute specification randomly from that list.
        # If only a single attribute is given, we use that.
        for attr in (set(mode_spec.keys()) - {'objs', 'tag'}):
            mode_spec[attr] = random_choice(mode_spec[attr])

        # Loop over objects.
        image_spec['objs'] = []
        for obj_spec in mode_spec['objs']:
            # In case list is given for an attribute, sample an attribute specification randomly from that list
            for attr in obj_spec.keys():
                obj_spec[attr] = random_choice(obj_spec[attr])

            obj = dict()

            # Object category / class.
            obj['category'] = obj_spec['category']
            obj['shape'] = obj_spec['shape']
            obj['texture'] = obj_spec['texture']


            if obj['category'] == 'test': # test datasets are correspondent
                obj['instance'] = self.iterables_test[obj['shape']].__next__()
            else: # train and val
                if self.train_val_sequential: # images in a sequence
                    if obj['category'] == 'train':
                        obj['instance'] = self.iterables_train[obj['shape']].__next__()
                    else: # val
                        obj['instance'] = self.iterables_val[obj['shape']].__next__()
                else: # random images (possible repetition)
                    last_instance_idx = DATASETS[obj['category']]['samples'][obj['shape']] 
                    numels_split = last_instance_idx // self.total_splits
                    obj['instance'] = np.random.randint(self.current_split*numels_split, (self.current_split+1)*numels_split)

            # Draw object color (hue + lightness).
            obj['color'] = sample_attribute('colorgrad',
                                            obj_spec['hue'],
                                            light_attr=obj_spec['lightness'])
            # Object position / scale.
            for attr in ['position', 'scale']:
                obj[attr] = sample_attribute(attr, obj_spec[attr])

            # Add object to sample.
            image_spec['objs'].append(obj)

        return image_spec, mode_spec
    
    def __len__(self):
        return len(self.permutation)

    def __getitem__(self, idx):
        idx = self.permutation[idx]
        return {
            'image': self.images[idx],
            'target': (self.task, self.task_labels[idx]),
            'tag': ''
        }

class DiagVib6DatasetPA(TorchDatasetWrapper):
    """Torch Dataset wrapper for DiagVib data in the PA framework. See description of Dataset and DatasetCSV_PA"""
    def __init__(self,
                 mnist_preprocessed_path: str,
                 dataset_specs_path: Optional[str] = None,
                 cache_filepath: Optional[str] = None,
                 t: str = 'train',
                 split_numsplit: Optional[List[int]] = [0, 1],
                 train_val_sequential: Optional[bool] = False,
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
                self.dataset = DatasetCSV_PA(
                    mnist_preprocessed_path=mnist_preprocessed_path,
                    csv_path=dataset_specs_path,
                    t=t,
                    split_numsplit=split_numsplit,
                    train_val_sequential=train_val_sequential,
                    seed=seed)
            else:
                self.dataset = Dataset(
                    dataset_spec=load_yaml(dataset_specs_path), 
                    mnist_preprocessed_path=mnist_preprocessed_path,
                    cache_path=None,
                    seed=seed + split_numsplit[0]  # to make them different
                )

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

    def __getitem__(self, idx: int):
        sample = self.dataset.__getitem__(idx)
        image, target, _ = sample.values()
        image = self._normalize(self._to_T(image, torch.float))
        return [image, self.unique_targets.index(target[1])] # we assume the task is the shape
    
class DiagVib6DatasetPABinary(DiagVib6DatasetPA):
    """
    Takes whatever the classes are, and generates a binary classification target.
    """
    def __getitem__(self, idx: int):
        sample = self.dataset.__getitem__(idx)
        image, target, _ = sample.values()
        image = self._normalize(self._to_T(image, torch.float))
        return [image, 2 * target[1] // len(self.unique_targets)] # we assume the task is the shape 
