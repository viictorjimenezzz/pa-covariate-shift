import numpy as np
import torch
from pandas import read_csv
from copy import deepcopy
import os
from PIL import Image
from torch.utils.data import Dataset


from src.data.components.diagvibsix.dataset.config import OBJECT_ATTRIBUTES

FACTORS = deepcopy(OBJECT_ATTRIBUTES)
FACTORS['position_factor'] = FACTORS.pop('position')
FACTORS['scale_factor'] = FACTORS.pop('scale')
#

class DiagVibSixDatasetSimple(Dataset):
    def __init__(self, root_dir, transform=None):
        super().__init__()

        self.metadata_fields = ['shape', 'hue', 'lightness', 'texture', 'position_factor', 'scale_factor']

        self.metadata = read_csv(os.path.join(root_dir, 'metada.csv'))
        self.transform = transform
        self.permutations = self.metadata.permutation.to_list()
        self.targets = self.metadata.task_labels.to_list()
        self.label_correction = {}
        for i, label in enumerate(np.unique(np.array(self.targets))):
            self.label_correction[label] = i
        self.label_recorrection = {v: k for k, v in self.label_correction.items()}
        self.targets = [self.label_correction[target] for target in self.targets]
        self.mean, self.std = np.genfromtxt(os.path.join(root_dir, 'mean_std.csv'), delimiter=',')
        self.mean = np.expand_dims(self.mean, axis=(1, 2))
        self.std = np.expand_dims(self.std, axis=(1, 2))

        images_dir = os.path.join(root_dir, 'images')
        names = os.listdir(images_dir)
        names = [name.split('.')[0] for name in names if name.endswith('jpeg')]
        names.sort(key=float)
        self.images_path = [os.path.join(images_dir, name + '.jpeg') for name in names]
        self.img_shape = [3, 32, 32]
    #     def _get_mt_labels(task_label):
    #         return np.argmax([cls == task_label[1] for cls in OBJECT_ATTRIBUTES[task_label[0]]])
    
    
    def _normalize(self, X):
        return X.sub_(torch.tensor(self.mean)).div_(torch.tensor(self.std))

    def _to_T(self, x, dtype):
        return torch.from_numpy(x).type(dtype)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        idx = self.permutations[idx]

        # Image
        img_path = self.images_path[idx]

        image = Image.open(img_path)
        image = np.moveaxis(np.array(image), 2, 0)
        image = self._normalize(self._to_T(image, torch.float))
        if self.transform:
            image = self.transform(image)

        #target = torch.tensor(_get_mt_labels(self.targets[idx]),dtype=torch.long)
        target = torch.tensor(self.targets[idx], dtype=torch.long)

        # metadata
        metadata = torch.tensor(self.metadata.iloc[idx, 2:8].to_list(), dtype=torch.float)
        # return [image, target, metadata]
        return [image,target]