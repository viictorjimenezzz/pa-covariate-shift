import torch
from secml.array import CArray


def carray2tensor(carr: CArray, dtype: torch.dtype):
    """ Converts a secml CArray to a PyTorch Tensor

        Inputs:
        carr (secml.CArray): the carray to convert 
        dtype (torch.dtype): the torch dtype of the resulting tensor
    """
    return torch.from_numpy(carr.tondarray()).to(dtype)
