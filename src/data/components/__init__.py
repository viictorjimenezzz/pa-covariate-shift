import torch
from torch import Tensor
from torch.utils.data import Dataset, TensorDataset, Subset, ConcatDataset
from typing import List, Optional

class MultienvDataset(Dataset):
    """
    We assume datasets return a tuple (input tensor, label).
    """

    def __init__(self, dset_list: List[Dataset]):
        len_ds = min([len(ds) for ds in dset_list])

        same_size = True
        for ds in dset_list:
            if len(ds) != len_ds:
                same_size = False
                break

        self.dset_list = dset_list
        self.num_envs = len(dset_list)
        self.permutation = [torch.arange(len(ds)).tolist() for ds in dset_list]

    def __len__(self):
        return min([len(perm) for perm in self.permutation])
 
    def __getitem__(self, idx: int):
        return {str(i): dset[self.permutation[i][idx]] for i, dset in enumerate(self.dset_list)}
    
    def __getitems__(self, indices: List[int]):
        """
        When I request several items, I prefer to get a tensor for each dataset.
        """
        # Is there a way to do it without multiplicating the calls to __getitem__?
        output_list = [None]*self.num_envs
        for i, dset in enumerate(self.dset_list):
            output_list[i] = tuple([torch.stack([self.__getitem__(idx)[str(i)][0] for idx in indices]), 
                                    torch.tensor([self.__getitem__(idx)[str(i)][1] for idx in indices])])

        return output_list
    
    def __getlabels__(self, indices: List[int]):
        """
        Useful method to retrieve only the labels associated with a specific index. This will help with the pairing of samples for the metric.
        """

        output_list = [None]*self.num_envs
        for i, dset in enumerate(self.dset_list):
            output_list[i] = torch.tensor([self.__getitem__(idx)[str(i)][1] for idx in indices]) 

        return output_list
    
    def Subset(self, indices: List[int]):
        """
        Returns a new MultienvDataset object with the subset of the original dataset.
        """
        subset_items = self.__getitems__(indices)
        return MultienvDataset([TensorDataset(*env_subset) for env_subset in subset_items])


class MultienvDatasetTest(MultienvDataset):
    """
    Subclass of `MultienvDataset` that concatenates the dataset list and returns the elements and also a
    `domain_tag` indicating the environment from which the observation came from.
    """

    def __init__(self, dset_list: List[Dataset]):
        super().__init__(dset_list)
        self.dset_list = [ConcatDataset(dset_list)]
        self.permutation = [torch.arange(len(self.dset_list[0])).tolist()]
        self.domain_tag = torch.cat([
            i*torch.ones(len(ds), dtype=int)
            for i, ds in enumerate(dset_list)
        ])

    def __getitem__(self, idx: int):
        return self.dset_list[0][self.permutation[0][idx]], self.domain_tag[self.permutation[0][idx]]


class LogitsDataset(Dataset):
    """
    TorchDataset wrapper for logits computation in the PA metric.
    """
    def __init__(self, logits: List[Tensor], y: Tensor) -> None:
        self.num_envs = len(logits)
        self._check_input(logits, y)
        self.logits = logits
        self.y = y

    def _check_input(self, logits: List[Tensor], y: Tensor) -> None:
        assert self.num_envs == len(logits), "Must add a logit for each environment"
        assert all(logits[0].size(0) == logit.size(0) for logit in logits), "Size mismatch between logits"
        assert all(y.size(0) == logit.size(0) for logit in logits), "Size mismatch between y and logits"

    def __len__(self):
        return self.logits[0].size(0)

    def __additem__(self, logits: List[Tensor], y: Tensor) -> None:
        """
        This method is slow, because it's concatenating tensors, so it should be avoided whenever possible.
        """
        self._check_input(logits, y)
        self.y = torch.cat([self.y, y])
        
        for i in range(self.num_envs):
            self.logits[i] = torch.cat([self.logits[i], logits[i]]) 

    def __getitem__(self, index: int):
        return {str(i): tuple([self.logits[i][index], self.y[index]]) for i in range(self.num_envs)}
    
    def __getitems__(self, indices: List[int]):
        """
        When I request several items, I prefer to get a tensor for each dataset.
        """
        return [tuple([self.logits[i][indices], self.y[indices]]) for i in range(self.num_envs)]