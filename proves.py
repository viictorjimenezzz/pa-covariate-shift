# TO DOWNLOAD:
# import requests
# from pathlib import Path

# # Use the correct raw URL of the JSON file
# url = "https://github.com/cleanlab/label-errors/raw/main/original_test_labels/imagenet_val_set_original_labels.npy"

# # Specify the destination folder and filename
# folder_path = Path("data/cleanlab/ImageNet/")
# folder_path.mkdir(parents=True, exist_ok=True)  # Ensure the folder exists

# filename = url.split("/")[-1]
# file_path = folder_path / filename

# # Download the file
# response = requests.get(url)
# response.raise_for_status()  # Ensure the download succeeded

# # Write the content to a file in the specified folder
# with open(file_path, 'wb') as file:
#     file.write(response.content)

# print(f"Downloaded '{file_path}' successfully.")
# exit()

from typing import Optional
import os.path as osp
import numpy as np
import json


from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, Subset, DataLoader

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

class CorrectedValidationImageNet(Dataset):
    def __init__(self, corrected: bool = True):
        """
        Provides the corrected validation ImageNet dataset.

        Args:
            corrected (bool): If True, returns the correct labels, otherwise it returns the original labels (i.e. mislabelled).

        The criterion to add an observation to the corrected validation (test) ImageNet dataset is the following:
            - The original label must be different from the predicted label.
            - The predicted label must be agreed by at least 3/5 mturk workers.
        """
        self.corrected = corrected

        index_val_path = r"data/cleanlab/ImageNet/imagenet_val_set_index_to_filepath.json"
        with open(index_val_path, 'r') as file:
            index_val = json.load(file) # list

        mturk_labels_path = r"data/cleanlab/ImageNet/imagenet_mturk.json"
        with open(mturk_labels_path, 'r') as file:
            mturk_labels = json.load(file)

        # len(index_val) = 50000
        # len(mturk_labels) = 5440

        # Names of all files in the order they appear in the dataset.
        index_names = [
            index_val_i.split("/")[-1]
            for index_val_i in index_val
        ] # 50000

        # Extract mturk names and labels: only if the guessed label has an agreement >=3/5 mturk workers.
        mturk_labels_filtered = [
            mturk_lab
            for mturk_lab in mturk_labels
            if mturk_lab['mturk']['guessed'] >= 3
        ]
        mturk_names = [
            mturk_lab['url'].split("/")[-1]
            for mturk_lab in mturk_labels_filtered
        ]

        # Then we can get the indexes of the mturk_names: this is our mask on the dataset
        index_corrected = np.array([
            index_names.index(mturk_name)
            for mturk_name in mturk_names
        ], dtype=int)

        # And the true labels associated to those indexes.
        self.corrected_labels = np.array([
            mturk_lab['our_guessed_label']
            for mturk_lab in mturk_labels_filtered
        ], dtype=int)

        # As well as the original labels:
        self.original_labels = np.array([
            mturk_lab['given_original_label']
            for mturk_lab in mturk_labels_filtered
        ], dtype=int)

        # Here I create the imagefolder, etc... all from my stored data.
        self.dataset = Subset(dataset, index_corrected)

    def __len__(self):
        assert self.corrected_labels == self.original_labels # TODO: REMOVE AFTER DEBUGGING
        return len(self.corrected_labels)

    def __getitem__(self, idx: int):
        x, y = self.dataset[idx]
        assert y == self.original_labels[idx] # TODO: REMOVE AFTER DEBUGGING
        if self.corrected:
            return x, self.corrected_labels[idx]
        else:
            return x, self.original_labels[idx]
        

class ClearlabImageNet(LightningDataModule):
    def __init__(
        self,
        corrected: Optional[bool] = True,
        batch_size: Optional[int] = 64,
        num_workers: Optional[int] = 0,
        pin_memory: Optional[bool] = True,
        ):
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

    def setup(self, stage: str = "fit"):
        if stage == "fit":
            # load the whole train dataset. No validation is performed.

        elif stage == "test":
            # Validation data is used as test.
            self.test_dataset = CorrectedValidationImageNet(corrected=self.hparams.corrected)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory
            # drop_last = False because we don't have so many observations and we don't intend to train any robust model.
            sampler=DistributedSampler(self.train_dataset, drop_last=False, shuffle=True) if dist.is_initialized() else DistributedSampler(self.train_dataset, drop_last=False, shuffle=True, num_replicas=1, rank=0)
        )

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory
            # drop_last = False because we don't have so many observations and we don't intend to train any robust model.
            sampler=DistributedSampler(self.test_dataset, drop_last=False, shuffle=True) if dist.is_initialized() else DistributedSampler(self.test_dataset, drop_last=False, shuffle=True, num_replicas=1, rank=0)
        )




import ipdb; ipdb.set_trace()

