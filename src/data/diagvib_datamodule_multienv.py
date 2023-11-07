from typing import Any, Dict, Optional, Callable

import os
import os.path as osp

from torch.utils.data import DataLoader

from pytorch_lightning import LightningDataModule
from diagvibsix.data.dataset.preprocess_mnist import get_processed_mnist

from src.data.components import MultienvDataset
from src.data.components.diagvib_dataset import DiagVib6Dataset, select_dataset_spec


class DiagVibDataModuleMultienv(LightningDataModule):
    """
    Generates a DataLoader object out of two DiagVib6 environments. The datasets can be loaded from cache (.pkl) or generated using configuration files (.csv or .yml). 
    Information about the format of the dataset configuration files can be found in the DiagVib6Dataset class.

    Args:
        num_envs: Number of environments that will be used.
        envs_name: Stem name of the environment. The .pkl or .csv/.yml files must be named "train_{envs_name}{num_env}" (and "val_{envs_name}{num_env}"), where num_env=range(num_envs).
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
        num_envs: int,
        envs_name: str,
        datasets_dir: str = osp.join(".", "data", "datasets"),
        mnist_preprocessed_path: str = osp.join(".", "data", "dg", "mnist_processed.npz"),
        collate_fn: Optional[Callable] = None,
        batch_size: Optional[int] = 64,
        num_workers: Optional[int] = 0,
        pin_memory: Optional[bool] = False,
    ):
        super().__init__()

        self.num_envs = num_envs
        self.envs_name = envs_name
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

        # Training dataset
        self.train_ds_list = []
        for env_count in range(self.num_envs):
            dataset_specs_path, cache_filepath = select_dataset_spec(dataset_dir=self.datasets_dir, dataset_name='train_' + self.envs_name + str(env_count+1))
            self.train_ds_list.append(DiagVib6Dataset(
                mnist_preprocessed_path = self.mnist_preprocessed_path,
                dataset_specs_path=dataset_specs_path,
                cache_filepath=cache_filepath,
                t='train'))
            
        self.train_ds = MultienvDataset(self.train_ds_list)

        # Validation dataset
        val_exists = True
        self.val_ds_list = []
        for env_count in range(self.num_envs):
            dataset_specs_path, cache_filepath = select_dataset_spec(dataset_dir=self.datasets_dir, dataset_name='val_' + self.envs_name + str(env_count+1))
            if dataset_specs_path == None and not os.path.exists(cache_filepath):
                """
                If there is no validation dataset, the datamodule will not yield error.
                Nevertheless, the callbacks for the training will have to be disabled. Use: --callbacks=none
                """
                print("\nNo configuration or .pkl file has been provided for validation.\n")
                val_exists = False
                break
            else:
                self.val_ds_list.append(DiagVib6Dataset(
                    mnist_preprocessed_path = self.mnist_preprocessed_path,
                    dataset_specs_path=dataset_specs_path,
                    cache_filepath=cache_filepath,
                    t='val'))
            
        if val_exists:
            self.val_ds = MultienvDataset(self.val_ds_list)
        else:
            self.val_ds = None

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=True,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        if self.val_ds is not None:
            return DataLoader(
                dataset=self.val_ds,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                drop_last=True,
                shuffle=False,
                collate_fn=self.collate_fn
            ) 

if __name__ == "__main__":
    LightningDataModule()