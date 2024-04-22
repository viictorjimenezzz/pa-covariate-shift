from pytorch_lightning.callbacks import Callback
from torchmetrics import Recall
import torch

class SensitivityDomains_Callback(Callback):

    def __init__(
            self,
            n_classes: int,
            n_domains_train: int,
            n_domains_val: int,
            n_domains_test: int
        ):
        super().__init__()

        self.n_classes = n_classes
        self.n_domains_train, self.n_domains_val, self.n_domains_test = n_domains_train, n_domains_val, n_domains_test
        
        _task = "multiclass" if n_classes > 2 else "binary"

        self.train_sensitivity_average = Recall(task=_task, num_classes=n_classes, average="macro")
        self.train_sensitivity = {
            f'sensitivity_{i}': Recall(task=_task, num_classes=n_classes, average="macro") 
            for i in range(n_domains_train)
        }
        
        self.val_sensitivity_average = Recall(task=_task, num_classes=n_classes, average="macro")
        self.val_sensitivity = {
            f'sensitivity_{i}': Recall(task=_task, num_classes=n_classes, average="macro") 
            for i in range(n_domains_val)
        }
        
        self.test_sensitivity_average = Recall(task=_task, num_classes=n_classes, average="macro")
        self.test_sensitivity = {
            f'sensitivity_{i}': Recall(task=_task, num_classes=n_classes, average="macro") 
            for i in range(n_domains_test)
        }

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
        

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        y, preds = outputs["targets"], outputs["preds"]

        metrics_dict = {
            'train/sensitivity_average': self.train_sensitivity_average.to(pl_module.device)(preds, y)
        }
        for env in batch.keys(): 
            assert int(env) in range(self.n_domains_train), f"Environment {env} not in range {self.n_domains_train}."
            if "domain_tag" in outputs.keys():
                mask = (outputs["domain_tag"] == int(env))
            else:
                mask = self._mask_each_domain(batch, int(env))

            if mask.sum().item() == 0:
                continue
            metrics_dict[f'train/sensitivity_{env}'] = self.train_sensitivity[f'sensitivity_{env}'].to(pl_module.device)(preds[mask], y[mask])

        pl_module.log_dict(metrics_dict, prog_bar=False, on_step=True, on_epoch=True, logger=True, sync_dist=True)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        y, preds = outputs["targets"], outputs["preds"]

        metrics_dict = {
            'val/sensitivity_average': self.val_sensitivity_average.to(pl_module.device)(preds, y)
        }
        for env in batch.keys(): 
            assert int(env) in range(self.n_domains_val), f"Environment {env} not in range {self.n_domains_val}."
            if "domain_tag" in outputs.keys():
                mask = (outputs["domain_tag"] == int(env))
            else:
                mask = self._mask_each_domain(batch, int(env))

            if mask.sum().item() == 0:
                continue
            metrics_dict[f'val/sensitivity_{env}'] = self.val_sensitivity[f'sensitivity_{env}'].to(pl_module.device)(preds[mask], y[mask])

        pl_module.log_dict(metrics_dict, prog_bar=False, on_step=True, on_epoch=True, logger=True, sync_dist=True)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        y, preds = outputs["targets"], outputs["preds"]
        metric_name = trainer.checkpoint_metric

        metrics_dict = {
            f'test/sensitivity_average_{metric_name}': self.test_sensitivity_average.to(pl_module.device)(preds, y)
        }

        for env in torch.unique(outputs["domain_tag"]): 
            assert int(env) in range(self.n_domains_test), f"Environment {env} not in range {self.n_domains_test}."
            if "domain_tag" in outputs.keys():
                mask = (outputs["domain_tag"] == int(env))
            else:
                mask = self._mask_each_domain(batch, int(env))

            if mask.sum().item() == 0:
                continue
            metrics_dict[f'test/sensitivity_{env}_{metric_name}'] = self.test_sensitivity[f'sensitivity_{env}'].to(pl_module.device)(preds[mask], y[mask])

        pl_module.log_dict(metrics_dict, prog_bar=False, on_step=True, on_epoch=True, logger=True, sync_dist=False) # SINGLE DEVICE
