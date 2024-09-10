from typing import Optional
import os
import torch
from torch.utils.data import Dataset
import tarfile
from PIL import Image
import io
from torchvision import transforms

import numpy as np
import json

from torch.utils.data import Dataset, Subset, TensorDataset


class ImageNetDataset(Dataset):
    def __init__(
            self,
            dataset_dir: str,
        ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.index, self.num_elements_per_label, self.order_labels = self._create_index(
            self.training if hasattr(self, "training") else None
        )

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    def _extract_label(self, filename: str, memberpath: str, idx: int):
        """
        The training data are renamed with their correct class using the `rename_imagenet.py` and `rename_imagenet.txt` files.
        """
        return int(filename.split("/")[-1].split(".")[0])

    def _tar_filepath_generator(self):
        for root, dirs, files in os.walk(self.dataset_dir):
            for file in files:
                if file.endswith('.tar'):
                    yield os.path.join(root, file)

    def _create_index(self, training: Optional[bool] = None):
        index = []
        training = training if training != None else True
        num_elements_per_label = np.zeros(1000, dtype=int)
        order_labels = [None]
        for tar_path in self._tar_filepath_generator():
            with tarfile.open(tar_path, "r") as tar:
                for member in tar.getmembers():
                    if member.isfile() and member.name.endswith('.JPEG'):
                        index.append((tar_path, member))
                        if training == True:
                            label = self._extract_label(tar_path, None, None)
                            if order_labels[-1] != label:
                                order_labels.append(label)
                            num_elements_per_label[label] += 1
        return index, num_elements_per_label, order_labels[1:]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        tar_path, member = self.index[idx]
        with tarfile.open(tar_path, "r") as tar:
            file = tar.extractfile(member)
            image = Image.open(io.BytesIO(file.read())).convert('RGB')
            if self.transform:
                image = self.transform(image)
        return image, self._extract_label(tar_path, member.path, idx)



class ImageNetDatasetValidation(ImageNetDataset):
    """
    Small subclass to make the label depend on the name of the .JPEG file and not the .tar file.
    """
    def __init__(self,
            dataset_dir: str,
        ):
        self.training = False
        super().__init__(dataset_dir)

        index_val_path = r"data/cleanlab/ImageNet/imagenet_val_set_index_to_filepath.json"
        with open(index_val_path, 'r') as file:
            index_validation = json.load(file) # list

        self.index_validation_names = [
            index_val_i.split("/")[-1]
            for index_val_i in index_validation
        ]
        
        self.index_validation_class = [
            index_val_i.split("/")[-2]
            for index_val_i in index_validation
        ]

        with open(r"data/cleanlab/rename_imagenet.txt", 'r') as file:
            self.class_to_label = [
                line.split()[0] for line in file
            ]
        with open(r"data/cleanlab/rename_imagenet.txt", 'r') as file:
            self.name_to_label = [
                line.split()[2].lower() for line in file
            ]
                
    def _extract_label(self, filename: str, memberpath: str, idx: int):
        return self.class_to_label.index(self.index_validation_class[self.index_validation_names.index(memberpath)])
    

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.components import MultienvDataset

class PairedImagenet(MultienvDataset):
    """
    Used to pair training and validation observations by label. We will iterate all validation samples
    and take the first sample of the training dataset with such label.
    """
    def __init__(
            self,
            train_dataset_dir: str,
            val_dataset_dir: str
        ):
        train_ds = ImageNetDataset(train_dataset_dir)
        self.val_ds = ImageNetDatasetValidation(val_dataset_dir)

        train_ds_start_label = np.concatenate([
            np.zeros(1, dtype=int),
            np.cumsum(train_ds.num_elements_per_label[train_ds.order_labels])[:-1]
        ])
        train_ds_current_index_label = np.zeros(1000, dtype=int)

        train_ds_paired_x = torch.empty((len(self.val_ds),) + self.val_ds[0][0].shape)
        train_ds_paired_y = torch.empty(len(self.val_ds))
        for idx, (_, label) in enumerate(iter(self.val_ds)):
            index_label_start = train_ds.order_labels.index(label)
            train_idx = train_ds_start_label[index_label_start] + train_ds_current_index_label[label] # starting index + current iteration
            train_ds_current_index_label[label] += 1 # add one so to not repeat samples

            # Make sure it's correct:
            train_image, train_label = train_ds.__getitem__(train_idx)
            assert label == train_label

            # Now we fill the vectors:
            train_ds_paired_x[idx, :, :, :] = train_image 
            train_ds_paired_y[idx] = label

        self.train_ds_paired = TensorDataset(train_ds_paired_x, train_ds_paired_y)

        # The dataset list is the original validation plus the appended training samples.
        super().__init__(dset_list=[self.val_ds, self.train_ds_paired])


class CorrectedValidationImageNet(Dataset):
    def __init__(
            self,
            dataset_dir: str,
            mislabelled: bool = False
        ):
        # https://image-net.org/challenges/LSVRC/2011/browse-synsets.php
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

        validation_ds = ImageNetDatasetValidation(os.path.join(self.dataset_dir, "val"))

        mturk_labels_path = r"data/cleanlab/ImageNet/imagenet_mturk.json"
        with open(mturk_labels_path, 'r') as file:
            mturk_labels = json.load(file)
        # len(mturk_labels) = 5440

        # Extract mturk names and labels: only if the guessed label has an agreement >=3/5 mturk workers.
        mturk_labels_filtered = [
            mturk_lab
            for mturk_lab in mturk_labels
            if mturk_lab['mturk']['guessed'] >= 3
        ] # 1428

        self.true_labels = []
        mturk_labels_filtered_coincide = []
        for mturk_label in mturk_labels_filtered:
            try:
                self.true_labels.append(
                    validation_ds.name_to_label.index(mturk_label['our_guessed_label_name'].lower().replace(" ", "_"))
                )
                mturk_labels_filtered_coincide.append(mturk_label)
            except:
                continue

        mturk_names = [
            mturk_lab['url'].split("/")[-1]
            for mturk_lab in mturk_labels_filtered_coincide
        ] # 1425
        self.original_labels = [
            validation_ds.class_to_label.index(
                validation_ds.index_validation_class[validation_ds.index_validation_names.index(mturk_name)]
            )
            for mturk_name in mturk_names
        ]


        member_paths = [
            member.path for _, member in validation_ds.index
        ]
       
        index_corrected = [
            member_paths.index(mturk_name)
            for mturk_name in mturk_names
        ]

        self.dataset = Subset(validation_ds, index_corrected)

    def __len__(self):
        return len(self.true_labels)

    def __getitem__(self, idx: int):
        x, y = self.dataset[idx]
        try:
            assert y == self.original_labels[idx] # TODO: REMOVE AFTER DEBUGGING
            print("NO ERROR")
        except:
            import ipdb; ipdb.set_trace()

        if self.mislabelled:
            return x, self.original_labels[idx]
        else:
            return x, self.true_labels[idx]
            