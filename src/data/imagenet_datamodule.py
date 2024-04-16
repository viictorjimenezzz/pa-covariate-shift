from typing import Optional
import os.path as osp
import numpy as np
import json


from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, Subset, DataLoader

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from src.data.components.imagenet_datasets import ImageNetDataset, ImageNetDatasetValidation, CorrectedValidationImageNet
        
class ClearlabImageNet(LightningDataModule):
    def __init__(
        self,
        dataset_dir: str,
        n_classes: int,
        corrected: Optional[bool] = True,
        corrected_mislabelled: Optional[bool] = False,
        batch_size: Optional[int] = 64,
        num_workers: Optional[int] = 0,
        pin_memory: Optional[bool] = True,
        ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Change or add as arguments if needed.
        self.index_val_path = r"data/cleanlab/ImageNet/imagenet_val_set_index_to_filepath.json"
        self.mturk_labels_path = r"data/cleanlab/ImageNet/imagenet_mturk.json"

    @property
    def num_classes(self):
        return 1000

    def prepare_data(self):
        assert osp.exists(self.index_val_path), "The index_val_path does not exist."
        assert osp.exists(self.mturk_labels_path), "The mturk_labels_path does not exist."
        return
    
    def _load_val_test_ds(self):
        """
        Generate the desired validation/test dataset. There are two possibilities:
            - The whole validation/test dataset is loaded.
            - Only the corrected subset is loaded. In that case, the user must select whether the true labels or the original
            labels are to be returned.
        """

        if self.hparams.corrected:
            return CorrectedValidationImageNet(self.hparams.dataset_dir, self.hparams.corrected_mislabelled)
        else:
            return ImageNetDatasetValidation(
                osp.join(self.hparams.dataset_dir, "val")
            )


    def setup(self, stage: str = "fit"):
        if stage == "fit":
            self.train_ds = ImageNetDataset(
                osp.join(self.hparams.dataset_dir, "train")
            )

            self.val_ds = self._load_val_test_ds()

        elif stage == "test":
            self.test_ds = self._load_val_test_ds()

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            # drop_last = False because we don't have so many observations and we don't intend to train any robust model.
            sampler=DistributedSampler(self.train_ds, drop_last=False, shuffle=True) if dist.is_initialized() else DistributedSampler(self.train_ds, drop_last=False, shuffle=True, num_replicas=1, rank=0)
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            # drop_last = False because we don't have so many observations and we don't intend to train any robust model.
            sampler=DistributedSampler(self.val_ds, drop_last=False, shuffle=True) if dist.is_initialized() else DistributedSampler(self.val_ds, drop_last=False, shuffle=True, num_replicas=1, rank=0)
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            # drop_last = False because we don't have so many observations and we don't intend to train any robust model.
            sampler=DistributedSampler(self.test_ds, drop_last=False, shuffle=True) if dist.is_initialized() else DistributedSampler(self.test_ds, drop_last=False, shuffle=True, num_replicas=1, rank=0)
        )

