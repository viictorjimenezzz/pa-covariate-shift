from typing import Callable, Optional, Union, List

from omegaconf import DictConfig, OmegaConf

import os.path as osp
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from pytorch_lightning import LightningDataModule
from pytorch_lightning.trainer.supporters import CombinedLoader

from wilds import get_dataset
from src.data.components.wilds_dataset import WILDSDatasetEnv
from src.data.components import MultienvDatasetTest

class WILDSDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        n_classes: int,
        train_config: Union[dict, DictConfig],
        val_config: Optional[Union[dict, DictConfig]] = None,
        test_config: Optional[Union[dict, DictConfig]] = None,
        transform: Optional[Callable] = None, # TODO: Figure out the transform thing
        dataset_dir: str = osp.join(".", "data", "datasets"),
        cache: bool = True,
        batch_size: int = 64,
        pin_memory: bool = True,
        num_workers: int = 0,
        multiple_trainloader_mode='max_size_cycle',
    ):

        super().__init__()
        self.save_hyperparameters(logger=False)

        # Select train, val and test configurations
        self.hparams.train_config = OmegaConf.to_container(train_config, resolve=True) if isinstance(train_config, DictConfig) else train_config
        if val_config is not None:
            self.hparams.val_config = OmegaConf.to_container(val_config, resolve=True) if isinstance(val_config, DictConfig) else val_config
        if test_config is not None:
            self.hparams.test_config = OmegaConf.to_container(test_config, resolve=True) if isinstance(test_config, DictConfig) else test_config

        self.train_dset_list, self.val_dset_list, self.test_dset_list = None, None, None
    
    @property
    def num_classes(self):
        return self.hparams.n_classes

    def prepare_data(self):
        # If the dataset does not exist or cache is set to False, download data
        if osp.exists(self.hparams.dataset_dir) == False or self.hparams.cache == False:
            get_dataset(
                dataset=self.hparams.dataset_name, 
                download=True, 
                unlabeled=False, 
                root_dir=self.hparams.dataset_dir
            )

    def setup(self, stage: Optional[str] = None):
        self.dataset = get_dataset(
                    dataset=self.hparams.dataset_name, 
                    download=False, 
                    unlabeled=False, 
                    root_dir=self.hparams.dataset_dir
                )
        
        if stage == "fit":
            self.num_train_envs = len(self.hparams.train_config.keys())
            self.train_dset_list = []
            for env in self.hparams.train_config.keys():
                env_dset = WILDSDatasetEnv(
                    dataset=self.dataset,
                    env_config=self.hparams.train_config[env],
                    transform=self.hparams.transform
                )
                self.train_dset_list.append(env_dset)
            
            self.val_dset_list = []
            if self.hparams.val_config is not None:
                self.num_val_envs = len(self.hparams.val_config.keys())
                for env in self.hparams.val_config.keys():
                    env_dset = WILDSDatasetEnv(
                        dataset=self.dataset,
                        env_config=self.hparams.val_config[env],
                        transform=self.hparams.transform
                    )
                    self.val_dset_list.append(env_dset)

        if stage == "test":
            self.test_dset_list = []
            if self.hparams.test_config is not None:
                self.num_test_envs = len(self.hparams.test_config.keys())
                for env in self.hparams.test_config.keys():
                    env_dset = WILDSDatasetEnv(
                        dataset=self.dataset,
                        env_config=self.hparams.test_config[env],
                        transform=self.hparams.transform
                    )
                    self.test_dset_list.append(env_dset)

                self.test_dset = MultienvDatasetTest(self.test_dset_list)
                
    def train_dataloader(self):
        # Dictionary of dataloaders for the training.
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
        We set `shuffle=True` because each dataset has different size, and we want that the probability of each sample to be selected
        within each dataset is the same.
        """
        # CombinedLoader is equivalent to a dictionary of DataLoaders.
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