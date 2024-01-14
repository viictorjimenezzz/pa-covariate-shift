from typing import List, Optional, Callable

import os
import os.path as osp

from torch.utils.data import DataLoader, Subset, ConcatDataset, SequentialSampler

from pytorch_lightning import LightningDataModule
from diagvibsix.data.dataset.preprocess_mnist import get_processed_mnist

from src.data.components import MultienvDataset
from src.data.components.collate_functions import MultiEnv_collate_fn
from src.data.components.diagvib_dataset import DiagVib6DatasetPA, select_dataset_spec
from src.pa_metric_torch import PosteriorAgreementSampler

import torch.distributed
from torch.utils.data.distributed import DistributedSampler

class DiagVibDataModuleMultienv(LightningDataModule):
    """
    Generates a DataLoader object out of two DiagVib6 environments. The datasets can be loaded from cache (.pkl) or generated using configuration files (.csv or .yml). 
    Information about the format of the dataset configuration files can be found in the DiagVib6Dataset class.

    Args:
        num_envs: Number of environments that will be used.
        envs_name: Stem name of the environment. The .pkl or .csv/.yml files must be named "train_{envs_name}{num_env}" (and "val_{envs_name}{num_env}"), where num_env=range(num_envs).
        shift_ratio: Ratio of samples to be shifted from the first to the second environment.
        datasets_dir: Path to the directory containing the dataset configuration files or the cache.
        disjoint_envs: Boolean indicating whether the environments are generated from disjoint sample sets or not.
        train_val_sequential (bool): Boolean to indicate whether train/val datasets will be formed by taking a different sample each time. If the number of requested images for a class is smaller than the total number of images for that class, data in the same epoch will be non-repetitive. Test datasets are built this way by default.
        mnist_preprocessed_path: Path to the preprocessed MNIST dataset. If not available, it will be generated there.
        collate_fn: Collate function to be used by the DataLoader. Each collate function adapts the output for a specific model.
        batch_size: Batch size.
        num_workers: Number of workers.
        pin_memory: Whether to pin memory.
    """
    def __init__(
        self,
        envs_index: List[int] = [1],
        envs_name: str = 'env',
        datasets_dir: str = osp.join(".", "data", "datasets"),
        disjoint_envs: bool = False,
        train_val_sequential: bool = False,
        mnist_preprocessed_path: str = osp.join(".", "data", "dg", "mnist_processed.npz"),
        collate_fn: Optional[Callable] = None,
        batch_size: Optional[int] = 64,
        num_workers: Optional[int] = 0,
        pin_memory: Optional[bool] = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_envs = len(envs_index)
        self.envs_index = envs_index
        self.envs_name = envs_name
        self.datasets_dir = datasets_dir
        self.mnist_preprocessed_path = mnist_preprocessed_path
        self.collate_fn = collate_fn

        self.train_val_sequential = train_val_sequential
        self.disjoint_envs = disjoint_envs

    @property
    def num_classes(self):
        return 10

    def prepare_data(self):
        """Checks if the MNIST data is available. If not, downloads and processes it."""
        
        _ = get_processed_mnist(osp.dirname(self.mnist_preprocessed_path) + os.sep)
        pass

    def setup(self, stage: Optional[str] = None):

        # TRAINING DATASET
        self.train_ds_list = []
        for env_count in range(self.num_envs):
            index = self.envs_index[env_count]
            if not self.disjoint_envs:
                split_numsplit = [0,1]
            else:
                split_numsplit = [env_count, self.num_envs]

            dataset_specs_path, cache_filepath = select_dataset_spec(dataset_dir=self.datasets_dir, dataset_name='train_' + self.envs_name + str(index))
            self.train_ds_list.append(
                DiagVib6DatasetPA(
                mnist_preprocessed_path = self.mnist_preprocessed_path,
                dataset_specs_path=dataset_specs_path,
                cache_filepath=cache_filepath,
                split_numsplit=split_numsplit,
                train_val_sequential=self.train_val_sequential,
                t='train')
                )
            
        self.train_ds = MultienvDataset(self.train_ds_list)

        # VALIDATION DATASET
        val_exists = True
        self.val_ds_list = []
        for env_count in range(self.num_envs):
            index = self.envs_index[env_count]
            if not self.disjoint_envs:
                split_numsplit = [0,1]
            else:
                split_numsplit = [env_count, self.num_envs]

            dataset_specs_path, cache_filepath = select_dataset_spec(dataset_dir=self.datasets_dir, dataset_name='val_' + self.envs_name + str(index))
            if dataset_specs_path == None and not os.path.exists(cache_filepath):
                """
                If there is no validation dataset, the datamodule will not yield error.
                Nevertheless, the callbacks for the training will have to be disabled. Use: --callbacks=none
                """
                print("\nNo configuration or .pkl file has been provided for validation.\n")
                val_exists = False
                break
            else:
                self.val_ds_list.append(DiagVib6DatasetPA(
                    mnist_preprocessed_path = self.mnist_preprocessed_path,
                    dataset_specs_path=dataset_specs_path,
                    cache_filepath=cache_filepath,
                    split_numsplit=split_numsplit,
                    train_val_sequential=self.train_val_sequential,
                    t='val'))
            
        if val_exists:
            self.val_ds = MultienvDataset(self.val_ds_list)
        else:
            self.val_ds = None

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=True,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        if self.val_ds is not None:
            return DataLoader(
                dataset=self.val_ds,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                drop_last=True,
                shuffle=False,
                collate_fn=self.collate_fn
            ) 
        

class DiagVibDataModulePA(DiagVibDataModuleMultienv):
    """
    See parent class for full information about the arguments. This subclass is used to generate the dataloader for the PA optimization, consisting of a single environment each time 
    """
    def __init__(self, 
                 shift_ratio: float = 1.0,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_envs = 2
        self.shift_ratio = shift_ratio
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None):
        dataset_specs_path, cache_filepath = select_dataset_spec(dataset_dir=self.datasets_dir, dataset_name= self.envs_name + str(self.envs_index[0]))
        self.test_ds1 = DiagVib6DatasetPA(
            mnist_preprocessed_path = self.mnist_preprocessed_path,
            dataset_specs_path=dataset_specs_path,
            cache_filepath=cache_filepath,
            t='test')
        
        dataset_specs_path, cache_filepath = select_dataset_spec(dataset_dir=self.datasets_dir, dataset_name= self.envs_name + str(self.envs_index[1]))
        self.test_ds2 = DiagVib6DatasetPA(
            mnist_preprocessed_path = self.mnist_preprocessed_path,
            dataset_specs_path=dataset_specs_path,
            cache_filepath=cache_filepath,
            t='test')
        
        self.test_ds2_shifted = self._apply_shift_ratio(self.test_ds1, self.test_ds2)
        self.test_pairedds = MultienvDataset([self.test_ds1, self.test_ds2_shifted])

    def _set_sampler(self):
        """
        I don't need to disable the shuffling in the DistributedSampler to get corresponding observations X and X', as these are paired in the
        collate function. Nevertheless, I want to control strictly the data that is used for the PA optimization so that I can compare with the metric.
        """

        ddp_init = torch.distributed.is_available() and torch.distributed.is_initialized() 
        if ddp_init:
            return PosteriorAgreementSampler(self.test_pairedds, shuffle=False, drop_last = True)
        else:
            return PosteriorAgreementSampler(self.test_pairedds, shuffle=False, drop_last = True, num_replicas=1, rank=0)
        
    def train_dataloader(self):
        return DataLoader(
                dataset=self.test_pairedds,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                collate_fn=MultiEnv_collate_fn,
                sampler=self._set_sampler()
            )
    
    def val_dataloader(self):
        return self.train_dataloader()

    def _apply_shift_ratio(self, ds1, ds2):
        """Generates the two-environment dataset adding (1-shift_ratio)*len(ds2) samples of ds1 to ds2.

        Args:
            ds1: First environment dataset.
            ds2: Second environment dataset.

        Returns:
            Concatenated dataset with the shifted samples.
        """
        size = len(ds1)
        if size != len(ds2):
            raise ValueError("Both test datasets must have the same size.")
        
        num_samples_1 = int(size*(1 - self.shift_ratio))

        sampled_1 = Subset(ds1, range(num_samples_1)) # first (1-SR)*size_1 samples are from ds1
        sampled_2 = Subset(ds2, range(num_samples_1, size)) # complete with last samples of ds2

        return ConcatDataset([sampled_1, sampled_2])
    

from src.data.components.logits_pa import LogitsPA

class DiagVibDataModulePAlogits(LogitsPA, DiagVibDataModulePA):
    def __init__(self, classifier: torch.nn.Module, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.classifier = classifier


if __name__ == "__main__":
    LightningDataModule()


# EXAMPLE OF USE:
"""
import numpy as np
import matplotlib.pyplot as plt

dm = DiagVibDataModuleMultienv(
    envs_index = [0, 1],
    envs_name = 'bal',
    datasets_dir = "/cluster/home/vjimenez/adv_pa_new/data/dg/dg_datasets/submission/toy_dataset/",
    disjoint_envs = True,
    train_val_sequential = True,
    mnist_preprocessed_path="data/dg/mnist_processed.npz",
    collate_fn = MultiEnv_collate_fn,
    batch_size = 5
)

dm.prepare_data()
dm.setup()
train_dl = iter(dm.train_dataloader())
output = train_dl.__next__()
print(output.keys())

for i in range(2):
    images = output[str(i)][0]
    targets = output[str(i)][1]
    for j in range(2):
        im = np.transpose(images[j], (1, 2, 0))
        target = targets[j]
        plt.imshow(im)
        plt.title(str(target))
        plt.savefig("/cluster/home/vjimenez/adv_pa_new/" + f"train_{i}_{j}.png")


dm = DiagVibDataModuleTestPA(
    envs_index = [0, 1],
    envs_name = 'bal',
    datasets_dir = "/cluster/home/vjimenez/adv_pa_new/data/dg/dg_datasets/submission/toy_dataset/",
    mnist_preprocessed_path="data/dg/mnist_processed.npz",
    collate_fn = MultiEnv_collate_fn,
    batch_size = 5
)

dm.prepare_data()
dm.setup()
test_dl = iter(dm.train_dataloader())
output = test_dl.__next__()

print(type(output))
print(output.keys())


import matplotlib.pyplot as plt

for i in range(2):
    images = output[str(i)][0]
    targets = output[str(i)][1]
    for j in range(2):
        im = np.transpose(images[j], (1, 2, 0))
        target = targets[j]
        plt.imshow(im)
        plt.title(str(target))
        plt.savefig("/cluster/home/vjimenez/adv_pa_new/" + f"test_{i}_{j}.png")
"""