from typing import Any, Dict, Optional

import os.path as osp

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from pytorch_lightning import LightningDataModule

from secml.data.loader import CDataLoaderCIFAR10
from secml.ml.classifiers import CClassifierPyTorch
from secml.adv.attacks import CAttack

from src.data.components import MultienvDataset
from src.data.components.cifar10_dataset import AdversarialCIFAR10Dataset
from src.data.utils import carray2tensor
from src.data.components.collate_functions import MultiEnv_collate_fn

class CIFAR10DataModule(LightningDataModule):

    def __init__(
        self,
        classifier: torch.nn.Module,
        attack: CAttack,
        batch_size: int = 64,
        adversarial_ratio: float = 1.0,
        dataset_dir: str = osp.join(".", "data", "datasets"),
        checkpoint_fname: str = "adversarial_data.pt",
        cache: bool = False,
        verbose: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        self.num_envs = 2 # original + attacked
        self.classifier = classifier # torch.nn.Module
        self.cclassifier = CClassifierPyTorch(
            model=classifier,
            input_shape=(3, 32, 32),
            pretrained=True,
            batch_size=batch_size,
        )
        self.cclassifier.name = classifier.name
        self.attack = attack(classifier=self.cclassifier)

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
                self.cclassifier,
                self.hparams.dataset_dir,
                self.checkpoint_fname,
                self.hparams.adversarial_ratio,
                self.hparams.verbose,
                self.hparams.cache,
            )

    def setup(self, stage: Optional[str] = None):

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
                self.cclassifier,
                self.hparams.dataset_dir,
                self.checkpoint_fname,
                self.hparams.adversarial_ratio,
                self.hparams.verbose,
                self.hparams.cache,
            )

            self.train_dset_list = [self.original_dset, self.adversarial_dset]

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
    

from src.data.components import MultienvDataset

class CIFAR10DatasetPA(MultienvDataset):
    """
    Dataset with original and adversarial samples that will be fed to the Posterior
    Agreement metric.
    """

    def __init__(
        self,
        classifier: torch.nn.Module,
        attack: CAttack,
        adversarial_ratio: float = 1.0,
        dataset_dir: str = osp.join(".", "data", "datasets"),
        checkpoint_fname: str = "adversarial_data.pt",
        cache: bool = True,
        ):

        cifar_dm = CIFAR10DataModule(
            classifier=classifier,
            attack=attack,
            adversarial_ratio=adversarial_ratio,
            dataset_dir=dataset_dir,
            checkpoint_fname=checkpoint_fname,
            cache=cache
        )
        
        # Since dataset is instantiated in a single process, I can call them here.
        cifar_dm.prepare_data()
        cifar_dm.setup()

        super().__init__(cifar_dm.train_dset_list)

