import torch
from torch import nn, argmax, optim
from torch.autograd import grad
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import LightningModule
from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.utilities.memory import garbage_collection_cuda

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
        self.save_hyperparameters(ignore=["net", "loss"])

    def compute_penalty(self, logits: torch.Tensor, y: torch.Tensor):
        """
        Computes the additional penalty term to achieve invariant representation.
        """
        dummy_w = torch.tensor(1.).to(self.device).requires_grad_()
        with torch.enable_grad():
            loss = self.loss(logits*dummy_w, y).to(self.device)
        gradient = grad(loss, [dummy_w], create_graph=True)[0]
        return gradient**2

    def training_step(self, batch: dict, batch_idx: int):
        loss = 0
        ys, preds = [], []
        for env in list(batch.keys()):
            x, y = batch[env]

            garbage_collection_cuda()
            logits = self.model(x)
            penalty = self.compute_penalty(logits, y)
            loss += self.loss(logits, y) + self.hparams.lamb*penalty
            
            ys.append(y)
            preds.append(argmax(logits, dim=1))

        return {
            "loss": loss,
            "logits": logits,
            "targets": torch.cat(ys, dim=0),
            "preds": torch.cat(preds, dim=0)
        }
    
    def validation_step(self, batch: dict, batch_idx: int):
        loss = 0
        ys, preds = [], []
        for env in list(batch.keys()):
            x, y = batch[env]

            garbage_collection_cuda()
            logits = self.model(x)
            penalty = self.compute_penalty(logits, y)
            loss += self.loss(logits, y) + self.hparams.lamb*penalty
            
            ys.append(y)
            preds.append(argmax(logits, dim=1))

        return {
            "loss": loss,
            "logits": logits,
            "targets": torch.cat(ys, dim=0),
            "preds": torch.cat(preds, dim=0)
        }
    
    def test_step(self, batch: dict, batch_idx: int):
        (x, y), domain_tag = batch

        garbage_collection_cuda()
        logits = self.model(x)
        # penalty = self.compute_penalty(logits, y)
        return {
            "loss": self.loss(logits, y), #+ self.hparams.lamb*penalty,
            "logits": logits,
            "targets": y,
            "preds": argmax(logits, dim=1),
            "domain_tag": domain_tag
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
