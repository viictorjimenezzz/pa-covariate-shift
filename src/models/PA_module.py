from typing import Optional, Any

import torch
import torch.nn as nn

import pytorch_lightning as pl

from src.models.components.post_agr import PosteriorAgreementKernel
from torchmetrics.classification.accuracy import Accuracy

class PosteriorAgreementModule(pl.LightningModule):
    """Optimization over the inverse temperature parameter of the Posterior Agreement kernel.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        beta0: float,
        num_classes: int,
        classifier: Optional[nn.Module] = nn.Identity(),
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.model = classifier if classifier else nn.Identity() # the default does not work for hydra
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        self.num_classes = num_classes
        self.kernel = PosteriorAgreementKernel(beta0=beta0)
        self.afr_true = Accuracy(task="multiclass", num_classes=self.num_classes) # FIX THIS LATER TOO
        self.afr_pred = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.acc_pa = Accuracy(task="multiclass", num_classes=self.num_classes)

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
            y_pred = torch.argmax(o1.data, 1)
            y_pred_adv = torch.argmax(o2.data, 1)
            y_true = train_batch[train_batch["envs"][0]][1]

            # Second, compute the AFR
            values = {
            "val/AFR pred": self.afr_pred(y_pred_adv, y_pred),
            "val/AFR true": self.afr_true(y_pred_adv, y_true),
            "val/acc_pa": self.acc_pa(torch.cat([y_pred, y_pred_adv]), torch.cat([y_true, y_true])),
            }
            self.log_dict(values, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)

        return loss
    
    def on_train_batch_end(self, out, batch, bidx):
        if self.trainer.is_last_batch: # assume is the training batch, otherwise I have to store it in a variable
            self.betas.append(self.kernel.beta.item())

    def on_validation_start(self):
        if self.trainer.is_last_batch:
            self.model.eval()
            self.kernel.reset()

    def validation_step(self, batch: Any, bidx: int):
        if self.trainer.is_last_batch: # last batch for the trainer
            env_names = batch["envs"]
            x1, x2 = batch[env_names[0]][0], batch[env_names[1]][0]
            o1, o2 = self.model(x1), self.model(x2)
            self.kernel.evaluate(self.betas[-1], o1, o2)

            if bidx == (self.trainer.num_val_batches[0] - 1):
                # Retrieve final logPA for the (subset) batches
                logPA = self.kernel.log_posterior()
                self.logPAs.append(logPA.item())

                # Log the metrics
                values = {
                    "val/beta": self.betas[-1],
                    "val/logPA": logPA,
                    "val/PA": torch.exp(logPA),
                }
                self.log_dict(values, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)

        return None # no need to use loss
        
    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(self.parameters())
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = PosteriorAgreementModule(None, None, None)
