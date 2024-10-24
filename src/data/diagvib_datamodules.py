from typing import List, Optional, Union

import os
import os.path as osp

from torch.utils.data import DataLoader

import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from pytorch_lightning import LightningDataModule
from pytorch_lightning.trainer.supporters import CombinedLoader

from diagvibsix.data.dataset.preprocess_mnist import get_processed_mnist

from src.data.components import MultienvDatasetTest
from src.data.components.diagvib_dataset import DiagVib6DatasetPA, select_dataset_spec


class DiagVibDataModuleMultienv(LightningDataModule):
    """
    Generates a DataLoader object out of two DiagVib6 environments. The datasets can be loaded from cache (.pkl) or generated using configuration files (.csv or .yml). 
    Information about the format of the dataset configuration files can be found in the DiagVib6Dataset class.

    Args:

    """
    def __init__(
        self,
        n_classes: int,
        envs_name: str = 'env',
        envs_index_train: Optional[List[int]] = [0],
        envs_index_val: Optional[List[int]] = [0],
        envs_index_test: Optional[List[int]] = [0],
        train_dataset_dir: str = osp.join(".", "data", "datasets"),
        val_dataset_dir: str = osp.join(".", "data", "datasets"),
        test_dataset_dir: str = osp.join(".", "data", "datasets"),
        dataset_class: Optional[Dataset] = DiagVib6DatasetPA, # passed as object, no instantiate
        train_disjoint_envs: bool = True,
        val_disjoint_envs: bool = False,
        train_val_sequential: bool = False,
        mnist_preprocessed_path: str = osp.join(".", "data", "dg", "mnist_processed.npz"),
        batch_size: Optional[int] = 64,
        num_workers: Optional[int] = 0,
        pin_memory: Optional[bool] = False,
        multiple_trainloader_mode: Optional[str] ='max_size_cycle',
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.hparams.num_envs_train = len(self.hparams.envs_index_train)
        self.hparams.num_envs_val = len(self.hparams.envs_index_val)
        self.hparams.num_envs_test = len(self.hparams.envs_index_test)

    @property
    def num_classes(self):
        return self.hparams.n_classes

    def prepare_data(self):
        """Checks if the MNIST data is available. If not, downloads and processes it."""
        
        _ = get_processed_mnist(osp.dirname(self.hparams.mnist_preprocessed_path) + os.sep)
        pass

    def setup(self, stage: Optional[str] = None):

        if stage == "fit":
            self.train_dset_list = []
            for env_count in range(self.hparams.num_envs_train):
                index = self.hparams.envs_index_train[env_count]
                if not self.hparams.train_disjoint_envs:
                    split_numsplit = [0,1]
                else:
                    split_numsplit = [env_count, self.hparams.num_envs_train]

                ds_name = 'train_' + self.hparams.envs_name + str(index)
                dataset_specs_path, cache_filepath = select_dataset_spec(dataset_dir=self.hparams.train_dataset_dir, dataset_name=ds_name)                
                self.train_dset_list.append(
                        self.hparams.dataset_class(
                            mnist_preprocessed_path = self.hparams.mnist_preprocessed_path,
                            dataset_specs_path=dataset_specs_path,
                            cache_filepath=cache_filepath,
                            split_numsplit=split_numsplit,
                            train_val_sequential=self.hparams.train_val_sequential,
                            t='train',
                        )
                )
                
            self.val_dset_list = []
            for env_count in range(self.hparams.num_envs_val):
                index = self.hparams.envs_index_val[env_count]
                if not self.hparams.val_disjoint_envs:
                    split_numsplit = [0,1]
                else:
                    split_numsplit = [env_count, self.hparams.num_envs_val]

                ds_name = 'val_' + self.hparams.envs_name + str(index)
                dataset_specs_path, cache_filepath = select_dataset_spec(dataset_dir=self.hparams.val_dataset_dir, dataset_name=ds_name)
                if dataset_specs_path == None and not os.path.exists(cache_filepath):
                    """
                    If there is no validation dataset, the datamodule will not yield error.
                    Nevertheless, the callbacks for the training will have to be disabled. Use: --callbacks=none
                    """
                    print("\nNo configuration or .pkl file has been provided for validation.\n")
                    break
                else:
                    self.val_dset_list.append(
                        self.hparams.dataset_class(
                            mnist_preprocessed_path = self.hparams.mnist_preprocessed_path,
                            dataset_specs_path=dataset_specs_path,
                            cache_filepath=cache_filepath,
                            split_numsplit=split_numsplit,
                            train_val_sequential=self.hparams.train_val_sequential,
                            t='val',
                        )
                    )

        else:
            self.test_dset_list = []
            for env_count in range(self.hparams.num_envs_test):
                index = self.hparams.envs_index_test[env_count]
                
                # Test environments always non-disjoint because we don't test them at the same time.
                # Metrics are provided for each of them, and we want each of them to have exactly the same
                # instantiations, with different shift.
                split_numsplit = [0, 1]

                ds_name = 'test_' + self.hparams.envs_name + str(index)
                dataset_specs_path, cache_filepath = select_dataset_spec(dataset_dir=self.hparams.test_dataset_dir, dataset_name=ds_name)

                # Apply shift ratio for the test:
                self.test_dset_list.append(
                   self.hparams.dataset_class(
                            mnist_preprocessed_path = self.hparams.mnist_preprocessed_path,
                            dataset_specs_path=dataset_specs_path,
                            cache_filepath=cache_filepath,
                            split_numsplit=split_numsplit,
                            train_val_sequential=self.hparams.train_val_sequential,
                            t='test',
                    )
                )
            self.test_dset = MultienvDatasetTest(self.test_dset_list)
                
    def train_dataloader(self):
        return {
                str(i): DataLoader(
                    dataset=ds,
                    batch_size=self.hparams.batch_size,
                    num_workers=self.hparams.num_workers,
                    pin_memory=self.hparams.pin_memory,
                    sampler = DistributedSampler(ds, drop_last=True, shuffle=True) if dist.is_initialized() else DistributedSampler(ds, drop_last=True, shuffle=True, num_replicas=1, rank=0),
                ) for i, ds in enumerate(self.train_dset_list)
        }

    def val_dataloader(self):
        """
        CombinedLoader is equivalent to a dictionary of DataLoaders.
        """
        if len(self.val_dset_list) > 0:
            return CombinedLoader({
                str(i): DataLoader(
                    dataset=ds,
                    batch_size=self.hparams.batch_size,
                    num_workers=self.hparams.num_workers,
                    pin_memory=self.hparams.pin_memory,
                    sampler = DistributedSampler(ds, drop_last=True, shuffle=False) if dist.is_initialized() else DistributedSampler(ds, drop_last=True, shuffle=False, num_replicas=1, rank=0),
                ) for i, ds in enumerate(self.val_dset_list)
            }, self.hparams.multiple_trainloader_mode)

    def test_dataloader(self):
        """
        Here we can set `shuffle=False`, because only one dataset is used.
        """
        return DataLoader(
                dataset=self.test_dset,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                sampler = DistributedSampler(self.test_dset, drop_last=False, shuffle=False, num_replicas=1, rank=0)
        )