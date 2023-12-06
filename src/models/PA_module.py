from typing import Optional, List, Any

import numpy as np
import matplotlib.pyplot as plt
from pytorch_lightning.utilities.types import STEP_OUTPUT

import torch
import torch.nn as nn

import pytorch_lightning as pl

from src.models.components.post_agr import PosteriorAgreementKernel
from torchmetrics.classification.accuracy import Accuracy

from src.data.components import LogitsDataset
from torch.utils.data import DataLoader, SequentialSampler

#delete later
from src.models.components.dg_backbone import DGBackbone
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
from PIL import Image

class PosteriorAgreementModule(pl.LightningModule):
    """Optimization over the inverse temperature parameter of the Posterior Agreement kernel.
    """

    def __init__(
        self,
        classifier: nn.Module,
        optimizer: torch.optim.Optimizer,
        beta0: float,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.model = classifier
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        self.num_classes = 2
        self.kernel = PosteriorAgreementKernel(beta0=beta0)
        self.afr_true = Accuracy(task="multiclass", num_classes=self.num_classes) # FIX THIS LATER TOO
        self.afr_pred = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.acc_pa = Accuracy(task="multiclass", num_classes=self.num_classes)

        self.batch_size = 64 # default, changed later
        self.logits_dataset = LogitsDataset(2) # two environments
        self.betas = []
        self.logPAs = []
    
    def model_step(self, batch: dict):
        # QUICK FIX for projection. TODO: implement a better version
        self.model.eval()
        self.kernel.beta.data.clamp_(min=0.0)
        self.kernel.reset()
        
        env_names = batch["envs"]
        x1, x2 = batch[env_names[0]][0], batch[env_names[1]][0]

        with torch.no_grad():
            o1, o2 = self.model(x1), self.model(x2)

        self.kernel(o1, o2)

        loss = -self.kernel.log_posterior()
        return o1, o2, loss
    
    def training_step(self, train_batch: Any, batch_idx: int):
        o1, o2, loss = self.model_step(train_batch)

        if self.current_epoch == 0:  # AFR does not change during the epochs
            if batch_idx == 0:
                self.batch_size = o1.shape[0] # set batch size
            y_pred = torch.argmax(o1.data, 1)
            y_pred_adv = torch.argmax(o2.data, 1)
            y_true = train_batch["0"][1]

            # First, store the logits as a dataset
            self.logits_dataset.__additem__([o1, o2], y_true)

            # Second, compute the AFR
            values = {
            "val/AFR pred": self.afr_pred(y_pred_adv, y_pred),
            "val/AFR true": self.afr_true(y_pred_adv, y_true),
            "val/acc_pa": self.acc_pa(torch.cat([y_pred, y_pred_adv]), torch.cat([y_true, y_true])),
            }
            self.log_dict(values, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)

        return loss

    def on_train_epoch_end(self):
        beta_last = self.kernel.beta.item()
        self.betas.append(beta_last)

        self.model.eval()
        self.kernel.beta.data.clamp_(min=0.0)
        self.kernel.reset()
        logits_dataloader = DataLoader(dataset=self.logits_dataset,
                                       batch_size=self.batch_size, # same as the data
                                       num_workers=0, # we won't create subprocesses inside a subprocess, and data is very light
                                       pin_memory=False, # only dense CPU tensors can be pinned

                                       # Important so that it matches with the input data.
                                       shuffle=False,
                                       drop_last = False,
                                       sampler=SequentialSampler(self.logits_dataset))
        
        for bidx, batch in enumerate(logits_dataloader):
            logits, y = batch
            self.kernel.evaluate(beta_last, logits[0], logits[1])
                
        # Retrieve final logPA for the (subset) batches
        logPA = self.kernel.log_posterior()
        self.logPAs.append(logPA.item())

        # Log the metrics
        values = {
            "val/beta": beta_last,
            "val/logPA": logPA,
            "val/PA": torch.exp(logPA),
        }
        self.log_dict(values, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        
    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(self.parameters())
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = PosteriorAgreementModule(nn.Identity(), None, None)
