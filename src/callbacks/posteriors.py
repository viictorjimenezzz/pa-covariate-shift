from typing import List

import os
import torch
import torch.nn.functional as F
import pickle
from pytorch_lightning.callbacks import Callback

class Posterior_Callback(Callback):
    """
    Stores the mean and std of the posterior distribution over a test dataset.
    """
    def __init__(
            self,
            n_classes: int,
            optimal_beta: int,
            algorithm_name: str,
            dataset_name: str
        ):
        super().__init__()
        self.algorithm_name, self.dataset_name = algorithm_name, dataset_name

        self.optimal_beta = optimal_beta
        self.num_x = 0
        self.stored_x, self.stored_xbeta = torch.zeros(n_classes).cpu(), torch.zeros(n_classes).cpu()
        self.stored_x2, self.stored_xbeta2 = torch.zeros(n_classes).cpu(), torch.zeros(n_classes).cpu()

    @staticmethod
    def softmax(logits):
        return F.softmax(logits, dim=-1)
    
    @staticmethod
    def gibbs(logits, beta):
        return F.softmax(logits*beta, dim=-1)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        logits = outputs["logits"].detach().cpu()

        self.num_x += logits.shape[0]
        self.stored_x += self.softmax(logits).sort(dim=-1, descending=True)[0].sum(dim=0)
        self.stored_x2 += self.softmax(logits).sort(dim=-1, descending=True)[0].pow(2).sum(dim=0)
        self.stored_xbeta += self.gibbs(logits, self.optimal_beta).sort(dim=-1, descending=True)[0].sum(dim=0)
        self.stored_xbeta2 += self.gibbs(logits, self.optimal_beta).sort(dim=-1, descending=True)[0].pow(2).sum(dim=0)    
        
    def on_test_epoch_end(self, trainer, pl_module):
        """
        Store the posteriors in a file.
        """
        stored_x_mean = self.stored_x/self.num_x
        stored_xbeta_mean = self.stored_xbeta/self.num_x

        results = {
            'stored_x_mean': stored_x_mean,
            'stored_x_std': torch.sqrt(self.stored_x2/self.num_x - stored_x_mean**2),

            'stored_xbeta_mean': stored_xbeta_mean,
            'stored_xbeta_std': torch.sqrt(self.stored_xbeta2/self.num_x - stored_xbeta_mean**2),
        }

        file_path = rf"/cluster/home/vjimenez/adv_pa_new/results/posteriors/{self.dataset_name}_{self.algorithm_name}.pkl"
        with open(file_path, 'wb') as f:
            pickle.dump(results, f)

