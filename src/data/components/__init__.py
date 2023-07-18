from torch.utils.data import Dataset


class PairDataset(Dataset):
    """Dataset to return pairs of Dataset items."""
    def __init__(self, dset1: Dataset, dset2: Dataset):
        if len(dset1) != len(dset2):
            raise ValueError("'dset1' and 'dset2' must have the same size.")
        self.dset1 = dset1
        self.dset2 = dset2

    def __len__(self):
        return len(self.dset1)
 
    def __getitem__(self, idx: int):
        return {"first": self.dset1[idx], "second": self.dset2[idx]}