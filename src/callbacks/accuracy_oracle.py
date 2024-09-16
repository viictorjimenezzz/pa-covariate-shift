from typing import Optional

import torch
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import Callback
from torchmetrics import Accuracy

class AccuracyOracle_Callback(Callback):
    """
    Computes accuracy on test domains.
    """
    def __init__(
            self,
            n_classes: int,
            n_domains_test: int,
            envs_index_test: list,
            top_k: Optional[int] = 1
        ):
        super().__init__()
        self.envs_index_test = envs_index_test

        self.n_classes = n_classes
        _task = "multiclass" if n_classes > 2 else "binary"

        self.test_acc = {
            f'acc_{i}': Accuracy(task=_task, num_classes=n_classes, average="macro", top_k=top_k) 
            for i in range(n_domains_test)
        }

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """
        Generate test datasets.
        """
        if stage == "test":
            return
        trainer.datamodule.hparams.num_envs_test = len(self.envs_index_test)
        trainer.datamodule.hparams.envs_index_test = self.envs_index_test
        trainer.datamodule.setup("test")

    def _mask_each_domain(self, batch: dict, env_index: int):
        """Returns a mask for the complete target and preds vectors corresponding to the current domain."""

        intervals = torch.cat((
            torch.tensor([0]),
            torch.cumsum(
                torch.tensor([len(batch[env][1]) for env in batch.keys()]),
                dim=0
            )
        ))

        len_total = intervals[-1]
        return (torch.arange(len_total) >= intervals[env_index]) & (torch.arange(len_total) < intervals[env_index+1])

    def _every_test_batch(self, pl_module: LightningModule, batch: dict, batch_idx: int):
        """
        Copied from AccuracyDomains_Callback.
        """
        # I move the data to the right device
        (x, y), domain_tag = pl_module._extract_test_batch(batch) 
        batch = (x.to(pl_module.device), y.to(pl_module.device)), domain_tag.to(pl_module.device)

        # Now I call the test step of the original model
        outputs = pl_module.test_step(batch, batch_idx)
        y, preds = outputs["targets"], outputs["preds"]

        metrics_dict = {}
        for env in torch.unique(outputs["domain_tag"]): 
            assert int(env) in range(len(self.envs_index_test)), f"Environment {env} not in range {len(self.envs_index_test)}."
            if "domain_tag" in outputs.keys():
                mask = (outputs["domain_tag"] == int(env))
            else:
                mask = self._mask_each_domain(batch, int(env))

            if mask.sum().item() == 0:
                continue
            metrics_dict[f'oracle/acc_{env}'] = self.test_acc[f'acc_{env}'].to(pl_module.device)(preds[mask], y[mask])
        
        return metrics_dict

    def on_validation_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch, batch_idx, dataloader_idx):
        if batch_idx == 0:
            pl_module.model.eval()
            dl = trainer.datamodule.test_dataloader()
            for batch_idx, batch in enumerate(dl):
                metrics_dict = self._every_test_batch(pl_module, batch, batch_idx)
                pl_module.log_dict(metrics_dict, prog_bar=False, on_step=True, on_epoch=True, logger=True, sync_dist=False)


class AccuracyOracle_Remove(Callback):
    def __init__(
            self,
            n_classes: int,
            top_k: Optional[int] = 1
        ):
        super().__init__()
        self.n_classes = n_classes
        _task = "multiclass" if n_classes > 2 else "binary"
        self.test_acc = Accuracy(task=_task, num_classes=n_classes, average="macro", top_k=top_k)

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """
        Generate test datasets.
        """
        if stage == "test":
            return
        trainer.datamodule.setup("test")

    def _every_test_batch(self, pl_module: LightningModule, batch: dict, batch_idx: int):
        """
        Copied from AccuracyDomains_Callback.
        """
        # I move the data to the right device
        (x, y), _ = pl_module._extract_test_batch(batch) 
        batch = (x.to(pl_module.device), y.to(pl_module.device))

        # Now I call the test step of the original model
        outputs = pl_module.test_step(batch, batch_idx)
        y, preds = outputs["targets"], outputs["preds"]

        metrics_dict = {}
        metrics_dict['oracle/acc'] = self.test_acc.to(pl_module.device)(preds, y)
        return metrics_dict

    def on_validation_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch, batch_idx, dataloader_idx):
        if batch_idx == 0:
            pl_module.model.eval()
            dl = trainer.datamodule.test_dataloader()
            for batch_idx, batch in enumerate(dl):
                metrics_dict = self._every_test_batch(pl_module, batch, batch_idx)
                pl_module.log_dict(metrics_dict, prog_bar=False, on_step=True, on_epoch=True, logger=True, sync_dist=False)