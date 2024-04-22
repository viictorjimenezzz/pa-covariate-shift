import os
import pandas as pd
import ast
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
