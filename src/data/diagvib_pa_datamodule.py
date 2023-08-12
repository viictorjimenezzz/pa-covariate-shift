from typing import Any, Dict, Optional

import os.path as osp

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset, ConcatDataset, Subset

from pytorch_lightning import LightningDataModule

from src.data.components import PairDataset
from src.data.components.diagvib_dataset import DiagVibSixDatasetSimple


class DiagvibPADataModule(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.

    A DataModule implements 6 key methods:
        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        ds1_env: str = 'train_env1',
        ds2_env: str = 'train_env2',
        shift_ratio: float = 1.0,
        batch_size: int = 64,
        data_dir: str = osp.join(".", "data", "datasets"),
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()
        
        self.dataset_dir = os.path.join(data_dir,'dataset')
        self.shift_ratio = shift_ratio
        self.ds1_env = ds1_env
        self.ds2_env = ds2_env
        self.save_hyperparameters()
    @property
    def num_classes(self):
        return 10

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        pass


    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        
        dset1 = DiagVibSixDatasetSimple(root_dir=os.path.join(self.dataset_dir,self.ds1_env))
        dset2 = DiagVibSixDatasetSimple(root_dir=os.path.join(self.dataset_dir,self.ds2_env))

        
        dset2_shifted = apply_shift_ratio_2(dset1,dset2,self.shift_ratio)
        self.paired_dset = PairDataset(dset1,dset2_shifted)

    def train_dataloader(self):
            
        def diagvib_collate_fn(batch:list):
            aux = {}
            for key in batch[0]:  # iterate over "first" and "second"
                aux[key] = [
                    torch.cat(
                        [b[key][0].reshape(1,*dset1.img_shape) for b in batch], dim=0
                    ),
                    torch.tensor([b[key][1] for b in batch]),
                ]
            return aux
            
        return DataLoader(dataset=self.paired_dset,
                    batch_size=self.hparams.batch_size,
                    num_workers=self.hparams.num_workers,
                    pin_memory=self.hparams.pin_memory,
                    drop_last=False,
                    shuffle=False,
                    collate_fn=diagvib_collate_fn)

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass
    
    

    def _apply_shift_ratio(dataset_A, dataset_B):
        shift_ratio = 1 - self.shift_ratio
        size_A = len(dataset_A)
        num_samples_A = int(size_A * shift_ratio)
        num_samples_B = size_A - num_samples_A

        sampled_A = Subset(dataset_A, range(num_samples_A))
        sampled_B = Subset(dataset_B, range(len(dataset_B) - num_samples_B, len(dataset_B)))

        final_dataset = ConcatDataset([sampled_A, sampled_B])

        return final_dataset


if __name__ == "__main__":
    LightningDataModule()
