import torch
from torch.utils.data import Dataset, Subset, ConcatDataset

def apply_shift_ratio(ref_ds: Dataset, env_ds: Dataset, shift_ratio: float = 1.0):
    """
    Modifies `env_ds` with (1-self.shift_ratio)% samples from `ref_ds`.
    """
    size = len(ref_ds)
    if size != len(env_ds):
        raise ValueError("Both test datasets must have the same size.")
    
    num_samples_1 = int(size*(1 - shift_ratio))
    sampled_1 = Subset(ref_ds, range(num_samples_1)) # first (1-SR)*size_1 samples are from ds1
    sampled_2 = Subset(env_ds, range(num_samples_1, size)) # complete with last samples of ds2
    return ConcatDataset([sampled_1, sampled_2])

class MislabelledMixin:
    """
    Mixin class that mislabels a dataset based on a given ratio.
    """
    def __init__(self, mislabelled_ratio: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mislabelled_ratio = mislabelled_ratio
        dataset_length = len(self)
        
        # Get indices to mislabel:
        num_to_mislabel = int(mislabelled_ratio * dataset_length)
        mislabelled_indices = torch.randperm(dataset_length)[:num_to_mislabel]
        self.mislabelled_indices = set(mislabelled_indices.tolist())  # faster lookups

    def __getitem__(self, index):
        data, label = super().__getitem__(index)
        if index in self.mislabelled_indices:
            return data, -1
        else:
            return data, label

    