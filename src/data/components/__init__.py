import torch
from torch import Tensor
from torch.utils.data import Dataset, TensorDataset
from typing import List, Tuple

class MultienvDataset(Dataset):
    """
    We assume datasets return a tuple (input tensor, label).
    """

    def __init__(self, dset_list: List[Dataset]):
        len_ds = len(dset_list[0])
        for ds in dset_list:
            if len(ds) != len_ds:
                raise ValueError("All datasets must have the same size.")
            
        self.dset_list = dset_list
        self.num_envs = len(dset_list)
        self.permutation = [torch.arange(len(self.dset_list[0])).tolist()]*self.num_envs

    def __len__(self):
        return len(self.permutation[0])
 
    def __getitem__(self, idx: int):
        return {str(i): dset[self.permutation[i][idx]] for i, dset in enumerate(self.dset_list)}
    
    def __getitems__(self, indices: List[int]):
        """
        When I request several items, I prefer to get a tensor for each dataset.
        """
        # Is there a way to do it without multiplicating the calls to __getitem__?
        output_list = [None]*self.num_envs
        for i, dset in enumerate(self.dset_list):
            output_list[i] = tuple([torch.stack([dset.__getitem__(self.permutation[i][idx])[0] for idx in indices]), 
                                    torch.tensor([dset.__getitem__(self.permutation[i][idx])[1] for idx in indices])])
        
        return output_list
    
    def Subset(self, indices: List[int]):
        """
        Returns a new MultienvDataset object with the subset of the original dataset.
        """
        subset_items = self.__getitems__(indices)
        return MultienvDataset([TensorDataset(*env_subset) for env_subset in subset_items])
    
    def __getlabels__(self, indices: List[int]):
        """
        Useful method to retrieve only the labels associated with a specific index. This will help with the pairing of samples for the metric.
        """

        output_list = [None]*self.num_envs
        for i, dset in enumerate(self.dset_list):
            output_list[i] = torch.tensor([dset.__getitem__(self.permutation[i][idx])[1] for idx in indices])
        
        return output_list

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

    def __len__(self):
        return self.logits[0].size(0)