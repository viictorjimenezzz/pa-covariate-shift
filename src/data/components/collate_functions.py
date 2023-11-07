import torch
from torch import Tensor
from typing import List


"""
This script contains collate functions for ERM, IRM and LISA for when data comes from different environments.
By having them here, the same datamodule can be used for different models.
"""

def diagvib_collate_fn(batch: List, img_shape: Tensor = None):
    aux = {}
    for key in batch[0]:  # iterate over "first" and "second"
        aux[key] = [
            torch.cat(
                [
                    b[key][0].reshape(1, *img_shape) # self.dset1.img_shape
                    for b in batch
                ],
                dim=0,
            ),
            torch.tensor([b[key][1] for b in batch]),
        ]

    # Save environments list
    aux["env"] = [env for env in batch[0].keys()]
    return aux

def IRM_collate_fn(batch: List):
    batch_dict = {}
    if not isinstance(batch[0], dict): # just one environment
        x = torch.stack([b[0] for b in batch])
        y = torch.tensor([b[1] for b in batch])
        batch_dict["0"] = [x,y]
        batch_dict["envs"] = ["0"]
    else:
        for env in batch[0]:
            batch_dict[env] = [
                torch.stack([b[env][0] for b in batch]),
                torch.tensor([b[env][1] for b in batch]),
            ]

        # Save environments list
        batch_dict["envs"] = [env for env in batch[0].keys()]

    return batch_dict
    


def ERM_collate_fn(batch: List):

    if not isinstance(batch[0], dict): # just one environment
        x = torch.stack([b[0] for b in batch])
        y = torch.tensor([b[1] for b in batch])

    else: # more than one environment
        x = torch.stack([b[env][0] for env in batch[0] for b in batch])
        y = torch.tensor([b[env][1] for env in batch[0] for b in batch])

    return x, y


def LISA_collate_fn(batch: List):
    """
    Prepares data for mixup and selective augmentation.
    """

    if not isinstance(batch[0], dict): # just one environment (only LISA-D)
        x = torch.stack([b[0] for b in batch])
        y = torch.tensor([b[1] for b in batch])
        envs = ["0"]*len(batch)

    else: # more than one environment
        x = torch.stack([b[env][0] for env in batch[0] for b in batch])
        y = torch.tensor([b[env][1] for env in batch[0] for b in batch])
        envs = [env for env in batch[0] for b in batch]

    return x, y, envs


        
