from typing import Any, Dict, Optional

import os.path as osp

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from pytorch_lightning import LightningDataModule

from secml.data.loader import CDataLoaderCIFAR10
from secml.ml.classifiers import CClassifierPyTorch
from secml.adv.attacks import CAttack


from src.data.components import MultienvDataset
from src.data.components.adv import AdversarialCIFAR10Dataset
from src.data.utils import carray2tensor
from src.data.components.collate_functions import MultiEnv_collate_fn


class CIFAR10DataModule(LightningDataModule):
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
        classifier: torch.nn.Module,
        attack: CAttack,
        batch_size: int = 64,
        adversarial_ratio: float = 1.0,
        data_dir: str = osp.join(".", "data", "datasets"),
        checkpoint_fname: str = "adversarial_data.pt",
        cache: bool = False,
        verbose: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        self.classifier = CClassifierPyTorch(
            model=classifier,
            input_shape=(3, 32, 32),
            pretrained=True,
            batch_size=batch_size,
        )
        self.classifier.name = classifier.name
        self.attack = attack(classifier=self.classifier)

        self.checkpoint_fname = self.attack.info + ".pt"

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["classifier"])

        self.original_dset: Optional[Dataset] = None
        self.adversarial_dset: Optional[Dataset] = None

    @property
    def num_classes(self):
        return 10

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        CDataLoaderCIFAR10().load(val_size=0)

        # create the adversarial dataset that will be subsequently used
        if self.hparams.cache:
            AdversarialCIFAR10Dataset(
                self.attack,
                self.classifier,
                self.hparams.data_dir,
                self.checkpoint_fname,
                self.hparams.adversarial_ratio,
                self.hparams.verbose,
                self.hparams.cache,
            )

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.original_dset and not self.adversarial_dset:
            _, ts = CDataLoaderCIFAR10().load(val_size=0)
            ts.X /= 255.0
            self.original_dset = TensorDataset(
                carray2tensor(ts.X, torch.float32).reshape(-1, 3, 32, 32),
                carray2tensor(ts.Y, torch.long),
            )

            self.adversarial_dset = AdversarialCIFAR10Dataset(
                self.attack,
                self.classifier,
                self.hparams.data_dir,
                self.checkpoint_fname,
                self.hparams.adversarial_ratio,
                self.hparams.verbose,
                self.hparams.cache,
            )

            # PairDataset
            self.paired_dset = MultienvDataset(
                [self.original_dset, self.adversarial_dset]
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.paired_dset,
            batch_size=self.hparams.batch_size,
            collate_fn=MultiEnv_collate_fn,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
    
    def val_dataloader(self):
        return self.train_dataloader()

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    LightningDataModule()
