import os
import torch
from torch.utils.data import Dataset
import tarfile
from PIL import Image
import io
from torchvision import transforms

import numpy as np
import json

from torch.utils.data import Dataset, Subset


class ImageNetDataset(Dataset):
    def __init__(
            self,
            dataset_dir: str,
        ):
        super().__init__()

        self.dataset_dir = dataset_dir
        self.label_to_index = {}
        self.index = self._create_index()

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    def _extract_label(self, filename: str):
        return filename.split('/')[0].split(".")[0]

    def _tar_filepath_generator(self):
        for root, dirs, files in os.walk(self.dataset_dir):
            for file in files:
                if file.endswith('.tar'):
                    self.label_to_index[self._extract_label(file)] = len(self.label_to_index)
                    yield os.path.join(root, file)

    def _create_index(self):
        index = []
        for tar_path in self._tar_filepath_generator():
            with tarfile.open(tar_path, "r") as tar:
                for member in tar.getmembers():
                    if member.isfile() and member.name.endswith('.JPEG'):
                        index.append((tar_path, member))
        return index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        tar_path, member = self.index[idx]
        with tarfile.open(tar_path, "r") as tar:
            file = tar.extractfile(member)
            image = Image.open(io.BytesIO(file.read())).convert('RGB')
            if self.transform:
                image = self.transform(image)
        
        return image, self.label_to_index[self._extract_label(os.path.join(tar_path, member.name))]
    

class ImageNetDatasetValidation(ImageNetDataset):
    """
    Small subclass to make the label depend on the name of the .JPEG file and not the .tar file.
    """

    def _extract_label(self, filename: str):
        return filename.split('/')[0].split(".")[0].split("_")[0]
    
    def _tar_filepath_generator(self):
        for root, dirs, files in os.walk(self.dataset_dir):
            for file in files:
                if file.endswith('.tar'):
                    yield os.path.join(root, file)
                elif file.endswith('.JPEG'):
                    self.label_to_index[self._extract_label(file)] = len(self.label_to_index)


class CorrectedValidationImageNet(Dataset):
    def __init__(
            self,
            dataset_dir: str,
            mislabelled: bool = False
        ):
        """
        Provides the corrected validation ImageNet dataset, which is the subset of the validation dataset that presents
        incorrect labels in its original version.

        Args:
            corrected (bool): If True, returns the correct labels, otherwise it returns the original labels (i.e. mislabelled).

        The criterion to add an observation to the corrected validation (test) ImageNet dataset is the following:
            - The original label must be different from the predicted label.
            - The predicted label must be agreed by at least 3/5 mturk workers.
        """
        super().__init__()
        self.dataset_dir = dataset_dir
        self.mislabelled = mislabelled

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
        self.true_labels = np.array([
            mturk_lab['our_guessed_label']
            for mturk_lab in mturk_labels_filtered
        ], dtype=int)

        # As well as the original labels:
        self.original_labels = np.array([
            mturk_lab['given_original_label']
            for mturk_lab in mturk_labels_filtered
        ], dtype=int)

        self.dataset = Subset(
            ImageNetDatasetValidation(os.path.join(self.dataset_dir, "val")),
            index_corrected
        )

    def __len__(self):
        assert len(self.true_labels) == len(self.original_labels) # TODO: REMOVE AFTER DEBUGGING
        return len(self.true_labels)

    def __getitem__(self, idx: int):
        x, y = self.dataset[idx]
        assert y == self.original_labels[idx] # TODO: REMOVE AFTER DEBUGGING
        if self.mislabelled:
            return x, self.original_labels[idx]
        else:
            return x, self.true_labels[idx]
            