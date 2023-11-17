from torch.utils.data import Dataset
from typing import List    

class MultienvDataset(Dataset):
    """Dataset to return tuples of Dataset items."""

    def __init__(self, dset_list: List[Dataset]):
        len_ds = len(dset_list[0])
        for ds in dset_list:
            if len(ds) != len_ds:
                raise ValueError("All datasets must have the same size.")
            
        self.dset_list = dset_list

    def __len__(self):
        return len(self.dset_list[0])
 
    def __getitem__(self, idx: int):
        return {str(i): dset[idx] for i, dset in enumerate(self.dset_list)}