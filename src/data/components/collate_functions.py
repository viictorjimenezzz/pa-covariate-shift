import torch
from torch import Tensor
from typing import List


def MultiEnv_collate_fn(batch: List):
    """
    Collate function for multi-environment datasets and multi-environment models.

    The output is of the form:

    batch_dict = {
        "envs": [env1_name, env2_name, ...],
        "env1_name": [x1, y1],
        "env2_name": [x2, y2],
        ...
    """

    batch_dict = {}
    for env in batch[0]:
        batch_dict[env] = [
            torch.stack([b[env][0] for b in batch]),
            torch.tensor([b[env][1] for b in batch]),
        ]

    batch_dict["envs"] = [env for env in batch[0].keys()]
    return batch_dict


def SingleEnv_collate_fn(batch: List):
    """Collate function for multi-environment datasets and single-environment models.
    
    The output is of the form:
    (Tensor[x_0_env1, ..., x_n_env1, x_0_env2, ..., x_n_env2, ...],
        Tensor[y_0_env1, ..., y_n_env1, y_0_env2, ..., y_n_env2, ...])
    """
    x = torch.stack([b[env][0] for env in batch[0] for b in batch])
    y = torch.tensor([b[env][1] for env in batch[0] for b in batch])
    return x, y