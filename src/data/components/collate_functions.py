import torch
from torch import Tensor
from typing import List


def MultiEnv_collate_fn(batch: List):
    """
    Collate function for multi-environment datasets and multi-environment models.

    The output is of the form:

    batch_dict = {
        "env1_name": [x1, y1],
        "env2_name": [x2, y2],
        ...
    """

    def _stack_y(batch, env):
        try:
            return torch.tensor([b[env][1] for b in batch])
        except:
            return torch.cat([b[env][1] for b in batch])


    batch_dict = {}
    for env in batch[0]:
        import ipdb; ipdb.set_trace()
        batch_dict[env] = [
            torch.stack([b[env][0] for b in batch]),
            _stack_y(batch, env),
        ]

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