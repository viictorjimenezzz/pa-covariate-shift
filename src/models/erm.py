from typing import Union
import torch
from torch import nn, optim
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import LightningModule
from pytorch_lightning.core.optimizer import LightningOptimizer
from torchmetrics import Accuracy, F1Score, Recall, Specificity, Precision

class ERM(LightningModule):
    """Vanilla ERM traning scheme for fitting a NN to the training data"""

    def __init__(
        self,
        n_classes: int,
        net: nn.Module,
        loss: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: DictConfig,
    ):
        super().__init__()

        self.model = net
        self.loss = loss
        self.save_hyperparameters(ignore=["net"])

    def _extract_batch(self, batch: Union[dict, tuple]):
        """
        The batch can come from either a multienvironment CombinedLoader or from a single environment DataLoader. That is
        equivalent to using a `MultiEnv_collate_fn` or a `SingleEnv_collate_fn`.
        """
        if isinstance(batch, dict):
            x = torch.cat([batch[env][0] for env in batch.keys()], dim=0)
            y = torch.cat([batch[env][1] for env in batch.keys()])
            return x, y
        else:
            return batch
    
    def training_step(self, batch: Union[dict, tuple], batch_idx: int):
        x, y = self._extract_batch(batch)

        logits = self.model(x)
        return {
            "loss": self.loss(input=logits, target=y),
            "logits": logits,
            "targets": y,
            "preds": torch.argmax(logits, dim=1)
        }

    def validation_step(self, batch: Union[dict, tuple], batch_idx: int):
        x, y = self._combined_loader_to_single(batch) if isinstance(batch, dict) else batch

        logits = self.model(x)
        return {
            "loss": self.loss(input=logits, target=y),
            "logits": logits,
            "targets": y,
            "preds": torch.argmax(logits, dim=1)
        }

    def test_step(self, batch: Union[dict, tuple], batch_idx: int):
        x, y = batch

        logits = self.model(x)
        return {
            "loss": self.loss(input=logits, target=y),
            "logits": logits,
            "targets": y,
            "preds": torch.argmax(logits, dim=1)
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
