from typing import Any, Dict, Optional, Callable

import os
import os.path as osp

from torch.utils.data import (
    DataLoader,
    ConcatDataset,
    Subset,
)

from pytorch_lightning import LightningDataModule
from diagvibsix.data.dataset.preprocess_mnist import get_processed_mnist

from src.data.components import PairDataset
from src.data.components.diagvib_dataset import DiagVib6Dataset, select_dataset_spec


class DiagVibDataModule2Envs(LightningDataModule):
    """
    Generates a DataLoader object out of two DiagVib6 environments. The datasets can be loaded from cache (.pkl) or generated using configuration files (.csv or .yml). 
    Information about the format of the dataset configuration files can be found in the DiagVib6Dataset class.

    Args:
        env1_name: Name of the first environment. The .pkl or .csv/.yml files must be named "train_{env1_name}" and/or "val_{env1_name}".
        env2_name: Name of the second environment. The .pkl or .csv/.yml files must be named "train_{env1_name}" and/or "val_{env1_name}".
        shift_ratio: Ratio of samples to be shifted from the first to the second environment.
        datasets_dir: Path to the directory containing the dataset configuration files or the cache.
        mnist_preprocessed_path: Path to the preprocessed MNIST dataset. If not available, it will be generated there.
        collate_fn: Collate function to be used by the DataLoader. Each collate function adapts the output for a specific model.
        batch_size: Batch size.
        num_workers: Number of workers.
        pin_memory: Whether to pin memory.
    """
    def __init__(
        self,
        env1_name: str,
        env2_name: str,
        shift_ratio: float = 1.0,
        datasets_dir: str = osp.join(".", "data", "datasets"),
        mnist_preprocessed_path: str = osp.join(".", "data", "dg", "mnist_processed.npz"),
        collate_fn: Optional[Callable] = None,
        batch_size: Optional[int] = 64,
        num_workers: Optional[int] = 0,
        pin_memory: Optional[bool] = False,
    ):
        super().__init__()

        self.shift_ratio = shift_ratio
        self.ds_env1 = env1_name
        self.ds_env2 = env2_name
        self.datasets_dir = datasets_dir
        self.mnist_preprocessed_path = mnist_preprocessed_path
        self.collate_fn = collate_fn
        self.save_hyperparameters()

    @property
    def num_classes(self):
        return 10

    def prepare_data(self):
        """Checks if the MNIST data is available. If not, downloads and processes it."""
        
        _ = get_processed_mnist(osp.dirname(self.mnist_preprocessed_path) + os.sep)
        pass

    def setup(self, stage: Optional[str] = None):

        if stage == "fit" or stage is None:
            dataset_specs_path, cache_filepath = select_dataset_spec(dataset_dir=self.datasets_dir, dataset_name='train_' + self.ds_env1)
            self.train_ds1 = DiagVib6Dataset(
                mnist_preprocessed_path = self.mnist_preprocessed_path,
                dataset_specs_path=dataset_specs_path,
                cache_filepath=cache_filepath,
                t='train')
            
            dataset_specs_path, cache_filepath = select_dataset_spec(dataset_dir=self.datasets_dir, dataset_name='train_' + self.ds_env2)
            self.train_ds2 = DiagVib6Dataset(
                mnist_preprocessed_path = self.mnist_preprocessed_path,
                dataset_specs_path=dataset_specs_path,
                cache_filepath=cache_filepath,
                t='train')
            
            self.train_ds2_shifted = self._apply_shift_ratio(self.train_ds1, self.train_ds2)
            self.train_pairedds = PairDataset(self.train_ds1, self.train_ds2_shifted)

        if stage == 'validate': # or stage == "fit": # uncomment to activate callbacks
            dataset_specs_path, cache_filepath = select_dataset_spec(dataset_dir=self.datasets_dir, dataset_name='val_' + self.ds_env1)
            self.val_ds1 = DiagVib6Dataset(
                mnist_preprocessed_path = self.mnist_preprocessed_path,
                dataset_specs_path=dataset_specs_path,
                cache_filepath=cache_filepath,
                t='val')
            
            dataset_specs_path, cache_filepath = select_dataset_spec(dataset_dir=self.datasets_dir, dataset_name='val_' + self.ds_env2)
            self.val_ds2 = DiagVib6Dataset(
                mnist_preprocessed_path = self.mnist_preprocessed_path,
                dataset_specs_path=dataset_specs_path,
                cache_filepath=cache_filepath,
                t='val')
            
            self.val_ds2_shifted = self._apply_shift_ratio(self.val_ds1, self.val_ds2)
            self.val_pairedds = PairDataset(self.val_ds1, self.val_ds2_shifted)


    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_pairedds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=False,
            shuffle=False,
            collate_fn=self.collate_fn,
        )


    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_pairedds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=False,
            shuffle=False,
            collate_fn=self.collate_fn
        ) 


    def _apply_shift_ratio(self, ds1, ds2):
        """Generates the two-environment dataset adding shift_ratio*len(ds2) samples of ds1 to ds2.

        Args:
            ds1: First environment dataset.
            ds2: Second environment dataset.

        Returns:
            Concatenated dataset with the shifted samples.
        """

        shift_ratio = 1 - self.shift_ratio
        size_1 = len(ds1)
        num_samples_1 = int(size_1 * shift_ratio)
        num_samples_2 = size_1 - num_samples_1

        sampled_A = Subset(ds1, range(num_samples_1))
        sampled_B = Subset(
            ds2, range(len(ds2) - num_samples_2, len(ds2))
        )

        return ConcatDataset([sampled_A, sampled_B])



if __name__ == "__main__":
    LightningDataModule()