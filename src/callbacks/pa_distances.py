from typing import Optional, List

import torch
import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, LightningModule
from pametric.lightning.callbacks import MeasureOutput_Callback, KL_Callback, Wasserstein_Callback, CosineSimilarity_Callback

class PAOutput_Callback(MeasureOutput_Callback):
    """
    Callback containing all the available metrics to characterize the output of the
    model at every epoch. By combining them we need to compute the output of the model only once.
    """

    def __init__(self, *args, **kwargs):
        """
        Add more metrics to the list if needed.
        """
        super().__init__(*args, **kwargs)
        self.metric_callback_list = [
            KL_Callback(),
            Wasserstein_Callback(),
            CosineSimilarity_Callback()
        ]

    def _compute_average(self, *args, **kwargs) -> torch.Tensor:
        self.average = False # at the beginning all false
        avg = super()._compute_average(*args, **kwargs)

        # Add the number of environments as it will be used later.
        for metric in self.metric_callback_list:
            metric.num_envs = self.num_envs
        return avg
    
    def _iterate_and_sum(self, dataloader: DataLoader, model_to_eval: nn.Module) -> torch.torch.Tensor:
        sum_val = torch.zeros((self.num_envs-1, len(self.metric_callback_list)))
        dataloader = self._reinstantiate_dataloader(dataloader)
        for _, batch in enumerate(dataloader):
            # Here depends wether the features have to be extracted or not
            output = [
                model_to_eval.forward(batch[e][0], self.output_features)
                for e in list(batch.keys())
            ]
            
            sum_val += torch.tensor([
                [
                    metric._metric(output[e], output[e+1])
                    for metric in self.metric_callback_list
                ]
                for e in range(self.num_envs-1)
            ]) 
        return sum_val
    
    def _log_average(self, average_val: torch.Tensor) -> None:
        dict_to_log = {}
        for mind, metric in enumerate(self.metric_callback_list):
            dict_to_log.update(
                metric._log_average(
                    average_val[:, mind] if metric.average == False else average_val[:, mind] / self.len_dataset,
                    metric_name = metric.metric_name,
                    log = False
                )
            )
               
        self.log_dict(dict_to_log, prog_bar=False, on_step=False, on_epoch=True, logger=True, sync_dist=True)