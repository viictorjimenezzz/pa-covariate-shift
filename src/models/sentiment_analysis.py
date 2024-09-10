from typing import Union, Optional
from pytorch_lightning import LightningModule
from pytorch_lightning.core.optimizer import LightningOptimizer

import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForSequenceClassification

class AutoModelForSequenceClassificationLogits:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        model = AutoModelForSequenceClassification.from_pretrained(*args, **kwargs)
        original_forward = model.forward
        model.forward = lambda *inputs, **kw: original_forward(*inputs, **kw).logits
        return model

class SentimentAnalysisModule(LightningModule):
    def __init__(
        self,
        n_classes: int,
        loss: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[DictConfig] = None
        ):
        super().__init__()
        self.save_hyperparameters()

        self.loss = loss
        self.model = AutoModelForSequenceClassificationLogits.from_pretrained(
            pretrained_model_name_or_path="distilbert-base-uncased",
            num_labels=n_classes
        )

    def _extract_batch(self, batch: Union[dict, tuple]):        
        if isinstance(batch, dict):
            return batch["input_ids"], batch["labels"]
        return batch

    def model_step(self,  batch: Union[dict, tuple], batch_idx: int):
        x, y = self._extract_batch(batch)

        logits = self.model(x)

        return {
            "loss": self.loss(input=logits, target=y),
            "logits": logits,
            "targets": y,
            "preds": torch.argmax(logits, dim=1)
        }

    def training_step(self, batch: Union[dict, tuple], batch_idx: int):
        return self.model_step(batch=batch, batch_idx=batch_idx)
    
    def validation_step(self, batch: Union[dict, tuple], batch_idx: int):
        return self.model_step(batch=batch, batch_idx=batch_idx)

    def test_step(self, batch: Union[dict, tuple], batch_idx: int):
        return self.model_step(batch=batch, batch_idx=batch_idx)

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
    