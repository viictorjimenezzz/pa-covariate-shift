from typing import Union, Optional
from pytorch_lightning import LightningModule
from pytorch_lightning.core.optimizer import LightningOptimizer

import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForSequenceClassification

class SentimentAnalysisModule(LightningModule):
    def __init__(
        self,
        n_classes: int,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[DictConfig] = None
        ):
        super().__init__()
        self.save_hyperparameters()

        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path="distilbert-base-uncased",
            num_labels=n_classes
        )

    def _extract_batch(self, batch: Union[dict, tuple]):
        import ipdb; ipdb.set_trace()
        
        if isinstance(batch, dict):
            return batch["input_ids"], batch["labels"]
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
        pass

    def test_step(self, batch: Union[dict, tuple], batch_idx: int):
        pass

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
    