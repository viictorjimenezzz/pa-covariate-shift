from pytorch_lightning import LightningModule, LightningDataModule
from pytorch_lightning.core.optimizer import LightningOptimizer
import torch
import os.path as osp
from torch import nn, argmax, optim
from torch.autograd import grad
from torchmetrics import Accuracy, F1Score, Recall, Specificity, Precision

# For the PA metric
from src.pa_metric_torch import PosteriorAgreement
from src.data.diagvib_datamodules import DiagVibDataModulePA
from src.data.components.collate_functions import MultiEnv_collate_fn
from copy import deepcopy

class IRM(LightningModule):
    """Invariant Risk Minimization (IRM) module."""

    def __init__(
        self,
        n_classes: int,
        net: nn.Module,
        loss: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: DictConfig,
        
        lamb: float = 1.0
    ):
        super().__init__()

        self.model = net
        self.loss = loss
        self.save_hyperparameters(ignore=["net"])

    def compute_penalty(self, logits, y):
        """
        Computes the additional penalty term to achieve invariant representation.
        """
        dummy_w = torch.tensor(1.).to(self.device).requires_grad_()
        with torch.enable_grad():
            loss = self.loss(logits*dummy_w, y).to(self.device)
        gradient = grad(loss, [dummy_w], create_graph=True)[0]
        return gradient**2

    def training_step(self, batch, batch_idx):
        loss = 0
        ys, preds = [], []
        for env in list(batch.keys()):
            x, y = batch[env]

            logits = self.model(x)
            penalty = self.compute_penalty(logits, y)
            loss += self.loss(logits, y) + self.lamb*penalty
            
            ys.append(y)
            preds.append(argmax(logits, dim=1))

        return {
            "loss": loss,
            "logits": logits,
            "targets": torch.cat(ys, dim=0),
            "preds": torch.cat(preds, dim=0)
        }
    
    def validation_step(self, batch, batch_idx):
        loss = 0
        ys, preds = [], []
        for env in list(batch.keys()):
            x, y = batch[env]

            logits = self.model(x)
            penalty = self.compute_penalty(logits, y)
            loss += self.loss(logits, y) + self.lamb*penalty
            
            ys.append(y)
            preds.append(argmax(logits, dim=1))

        return {
            "loss": loss,
            "logits": logits,
            "targets": torch.cat(ys, dim=0),
            "preds": torch.cat(preds, dim=0)
        }
    
    def test_step(self, batch, batch_idx):
        assert len(batch.keys()) == 1, "The test batch should have only one environment."
        x, y = batch

        logits = self.model(x)
        penalty = self.compute_penalty(logits, y)
        return {
            "loss": self.loss(logits, y) + self.lamb*penalty,
            "logits": logits,
            "targets": y,
            "preds": argmax(logits, dim=1)
        }
            
    def configure_optimizers(self):
        optimizer = LightningOptimizer(self.hparams.optimizer(params=self.parameters()))

        if self.hparams.scheduler:
            scheduler = self.hparams.scheduler.scheduler(optimizer=optimizer)

            scheduler_dict = OmegaConf.to_container(self.hparams.scheduler, resolve=True) # convert to normal dict
            scheduler_dict.update({
                    "scheduler": scheduler,
            })

            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler_dict
            }
        
        return {"optimizer": optimizer}
