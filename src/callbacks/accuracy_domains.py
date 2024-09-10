from typing import Optional

import torch
from pytorch_lightning.callbacks import Callback
from torchmetrics import Accuracy

class AccuracyDomains_Callback(Callback):
    """
    Computes and logs general accuracy metrics specific for domain shifts (OOD and subpopulation shifts):
        - Average accuracy across all domains.
        - Worst domain accuracy.
    """
    def __init__(
            self,
            n_classes: int,
            top_k: Optional[int] = 1,
            n_domains_train: Optional[int] = None,
            n_domains_val: Optional[int] = None,
            n_domains_test: Optional[int] = None
        ):
        super().__init__()

        self.n_classes = n_classes
        self.n_domains_train, self.n_domains_val, self.n_domains_test = n_domains_train, n_domains_val, n_domains_test
        
        _task = "multiclass" if n_classes > 2 else "binary"

        if self.n_domains_train is not None:
            self.train_acc_average = Accuracy(task=_task, num_classes=n_classes, average="macro", top_k=top_k)
            self.train_acc = {
                f'acc_{i}': Accuracy(task=_task, num_classes=n_classes, average="macro", top_k=top_k) 
                for i in range(n_domains_train)
            }
        
        if self.n_domains_val is not None:
            self.val_acc_average = Accuracy(task=_task, num_classes=n_classes, average="macro", top_k=top_k)
            self.val_acc = {
                f'acc_{i}': Accuracy(task=_task, num_classes=n_classes, average="macro", top_k=top_k) 
                for i in range(n_domains_val)
            }
        
        if self.n_domains_test is not None:
            self.test_acc_average = Accuracy(task=_task, num_classes=n_classes, average="macro", top_k=top_k)
            self.test_acc = {
                f'acc_{i}': Accuracy(task=_task, num_classes=n_classes, average="macro", top_k=top_k) 
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
        if self.n_domains_train is None:
            return

        y, preds = outputs["targets"], outputs["preds"]

        metrics_dict = {
            'train/acc_average': self.train_acc_average.to(pl_module.device)(preds, y)
        }
        for env in batch.keys(): 
            assert int(env) in range(self.n_domains_train), f"Environment {env} not in range {self.n_domains_train}."
            if "domain_tag" in outputs.keys():
                mask = (outputs["domain_tag"] == int(env))
            else:
                mask = self._mask_each_domain(batch, int(env))

            if mask.sum().item() == 0:
                continue
            metrics_dict[f'train/acc_{env}'] = self.train_acc[f'acc_{env}'].to(pl_module.device)(preds[mask], y[mask])

        pl_module.log_dict(metrics_dict, prog_bar=False, on_step=True, on_epoch=True, logger=True, sync_dist=True)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if self.n_domains_val is None:
            return
        
        y, preds = outputs["targets"], outputs["preds"]

        metrics_dict = {
            'val/acc_average': self.val_acc_average.to(pl_module.device)(preds, y)
        }
        for env in batch.keys(): 
            assert int(env) in range(self.n_domains_val), f"Environment {env} not in range {self.n_domains_val}."
            if "domain_tag" in outputs.keys():
                mask = (outputs["domain_tag"] == int(env))
            else:
                mask = self._mask_each_domain(batch, int(env))

            if mask.sum().item() == 0:
                continue
            metrics_dict[f'val/acc_{env}'] = self.val_acc[f'acc_{env}'].to(pl_module.device)(preds[mask], y[mask])

        pl_module.log_dict(metrics_dict, prog_bar=False, on_step=True, on_epoch=True, logger=True, sync_dist=True)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if self.n_domains_test is None:
            return
        
        y, preds = outputs["targets"], outputs["preds"]
        metric_name = trainer.checkpoint_metric

        metrics_dict = {
            f'test/acc_average_{metric_name}': self.test_acc_average.to(pl_module.device)(preds, y)
        }

        for env in torch.unique(outputs["domain_tag"]): 
            assert int(env) in range(self.n_domains_test), f"Environment {env} not in range {self.n_domains_test}."
            if "domain_tag" in outputs.keys():
                mask = (outputs["domain_tag"] == int(env))
            else:
                mask = self._mask_each_domain(batch, int(env))

            if mask.sum().item() == 0:
                continue
            metrics_dict[f'test/acc_{env}_{metric_name}'] = self.test_acc[f'acc_{env}'].to(pl_module.device)(preds[mask], y[mask])

        pl_module.log_dict(metrics_dict, prog_bar=False, on_step=True, on_epoch=True, logger=True, sync_dist=False) # SINGLE DEVICE
