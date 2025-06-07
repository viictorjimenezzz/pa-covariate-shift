from typing import Optional
import os

import torch
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.trainer.supporters import CombinedLoader
from pytorch_lightning import LightningDataModule

from secml.data.loader import CDataLoaderCIFAR10
from src.data.utils import carray2tensor
from src.data.components.cifar10aa import AdversarialCIFAR10DatasetAA


class CIFAR10DataModuleAA(LightningDataModule):
    """PyTorch Lightning DataModule for CIFAR-10 with AutoAttack adversarial examples."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        attack: str = None,  # e.g. 'apgd-ce', 'apgd-t'
        norm: str = 'Linf',
        eps: float = 8/255,
        version: str = 'standard',
        batch_size: int = 64,
        adversarial_ratio: float = 1.0,
        small_magnitude_first: bool = False,
        dataset_dir: str = os.path.join(".", "data", "datasets"),
        cache: bool = False,
        verbose: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs
    ):
        super().__init__()
        self.num_envs = 2
        self.model = model

        os.makedirs(dataset_dir, exist_ok=True)
        self.checkpoint_fname = f"aa={attack}_eps={eps}_version={version}.pt"

        self.save_hyperparameters(logger=False, ignore=["model"])
    
    @property
    def num_classes(self):
        return 10
    
    def prepare_data(self):
        """Download data if needed. Do not use it to assign state (self.x = y)."""
        CDataLoaderCIFAR10().load(val_size=0)
        
        # Pre-generate adversarial dataset if caching is enabled
        if self.hparams.cache:
            AdversarialCIFAR10DatasetAA(
                model=self.model,
                attack=self.hparams.attack,
                norm=self.hparams.norm,
                eps=self.hparams.eps,
                version=self.hparams.version,
                data_dir=self.hparams.dataset_dir,
                checkpoint_fname=self.checkpoint_fname,
                adversarial_ratio=self.hparams.adversarial_ratio,
                small_magnitude_first=self.hparams.small_magnitude_first,
                verbose=self.hparams.verbose,
                cache=self.hparams.cache,
            )
    
    def setup(self, stage: Optional[str] = None):
        """Load and split datasets."""

        # Load original CIFAR-10 test set
        _, ts = CDataLoaderCIFAR10().load(val_size=0)
        ts.X /= 255.0
        self.original_dset = TensorDataset(
            carray2tensor(ts.X, torch.float32).reshape(-1, 3, 32, 32),
            carray2tensor(ts.Y, torch.long),
        )
        
        # Load/generate adversarial dataset
        self.adversarial_dset = AdversarialCIFAR10DatasetAA(
            model=self.model,
            attack=self.hparams.attack,
            norm=self.hparams.norm,
            eps=self.hparams.eps,
            version=self.hparams.version,
            data_dir=self.hparams.dataset_dir,
            checkpoint_fname=self.checkpoint_fname,
            adversarial_ratio=self.hparams.adversarial_ratio,
            verbose=self.hparams.verbose,
            cache=self.hparams.cache,
        )
        self.train_dset_list = [self.original_dset, self.adversarial_dset]
    
    def test_dataloader(self):
        return CombinedLoader(
            [
                DataLoader(
                    dataset=ds,
                    batch_size=self.hparams.batch_size,
                    num_workers=self.hparams.num_workers,
                    pin_memory=self.hparams.pin_memory,
                    shuffle=False
                )
                for ds in self.train_dset_list
            ],
            mode="min_size"
        )


from src.data.components import MultienvDataset

class CIFAR10DatasetPAAA(MultienvDataset):
    """Dataset with paired original and adversarial samples."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        attack: str = None,  # e.g. 'apgd-ce', 'apgd-t'
        norm: str = 'Linf',
        eps: float = 8/255,
        version: str = 'standard',
        adversarial_ratio: float = 1.0,
        dataset_dir: str = os.path.join(".", "data", "datasets"),
        checkpoint_fname: str = "autoattack_data.pt",
        cache: bool = True,
    ):
        
        cifar_dm = CIFAR10DataModuleAA(
            model=model,
            attack=attack,
            norm=norm,
            eps=eps,
            version=version,
            adversarial_ratio=adversarial_ratio,
            dataset_dir=dataset_dir,
            checkpoint_fname=checkpoint_fname,
            cache=cache,
        )
        
        # Since dataset is instantiated in a single process, we can call them here
        cifar_dm.prepare_data()
        cifar_dm.setup()
        
        super().__init__(cifar_dm.train_dset_list)