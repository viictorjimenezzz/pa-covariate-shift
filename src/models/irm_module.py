from pytorch_lightning import LightningModule, LightningDataModule
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
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler,
        lamb: float = 1.0
    ):
        super().__init__()

        self.model = None
        self.loss = None
        self.lamb = lamb
        self.save_hyperparameters(ignore=["net"])  # for easier retrieval from w&b and sanity checks

        self.n_classes = int(n_classes)
        _task = "multiclass" if self.n_classes > 2 else "binary"

        # Training metrics
        self.train_acc = Accuracy(task=_task, num_classes=self.n_classes, average="macro")
        self.train_f1 = F1Score(task=_task, num_classes=self.n_classes, average="macro")
        self.train_specificity = Specificity(task=_task, num_classes=self.n_classes, average="macro")
        self.train_sensitivity = Recall(task=_task, num_classes=self.n_classes, average="macro")
        self.train_precision = Precision(task=_task, num_classes=self.n_classes, average="macro")

        # Validation metrics
        self.val_acc = Accuracy(task=_task, num_classes=self.n_classes, average="macro")
        self.val_f1 = F1Score(task=_task, num_classes=self.n_classes, average="macro")
        self.val_specificity = Specificity(task=_task, num_classes=self.n_classes, average="macro")
        self.val_sensitivity = Recall(task=_task, num_classes=self.n_classes, average="macro")
        self.val_precision = Precision(task=_task, num_classes=self.n_classes, average="macro")

        # PA metric
        # TO DELETE
        # dm = DiagVibDataModuleTestPA(
        #     envs_index = [0, 1],
        #     shift_ratio = 1.0,
        #     envs_name = "val_randnobal", # here the full name not only the environment, as we may want to use test_ or val_ or even a custom name
        #     datasets_dir = "./data/dg/dg_datasets/randnobal/",
        #     mnist_preprocessed_path = "./data/dg/mnist_processed.npz",
        #     batch_size = 64,
        #     num_workers = 2,
        #     pin_memory = True,
        #     collate_fn = MultiEnv_collate_fn)
        
        # dm.prepare_data()
        # dm.setup()

        # self.PA = PosteriorAgreement(
        #             dataset = dm.test_pairedds,
        #             pa_epochs = 10,
        #             strategy = "lightning")

    def compute_penalty(self, logits, y):
        dummy_w = torch.tensor(1.).to(self.device).requires_grad_()
        with torch.enable_grad():
            loss = self.loss(logits*dummy_w, y).to(self.device)
        gradient = grad(loss, [dummy_w], create_graph=True)[0]
        return gradient**2

    def training_step(self, batch, batch_idx):
        envs = batch["envs"]
        outputs = {} 
        for env in envs:
            x, y = batch[env]
            logits = self.model(x)

            outputs[env] = {
                "logits": logits,
                "targets": y,
                "penalty": self.compute_penalty(logits, y)
            }
        
        return outputs

    def training_step_end(self, outputs):
        # Separate the training_step bc we want to sum losses from different GPUs here.
        loss = 0
        preds = []
        ys = []
        for env, output in outputs.items():
            logits = output["logits"]
            y = output["targets"]
            penalty = output["penalty"]

            env_loss = self.loss(logits, y)
            loss += env_loss + self.lamb*penalty
            
            ys.append(y)
            preds.append(argmax(logits, dim=1))
        y = torch.cat(ys, dim=0)
        preds = torch.cat(preds, dim=0)

        # Log training metrics
        metrics_dict = {
            "train/loss": loss,
            "train/acc": self.train_acc(preds, y),
            "train/f1": self.train_f1(preds, y),
            "train/sensitivity": self.train_sensitivity(preds, y),
            "train/specificity": self.train_specificity(preds, y),
            "train/precision": self.train_precision(preds, y),
        }
        self.log_dict(metrics_dict, prog_bar=False, on_step=True, on_epoch=True, logger=True, sync_dist=True)

        # Return loss for optimization
        return {"loss": loss}

    
    def validation_step(self, batch, batch_idx):
        envs = batch["envs"]

        outputs = {}
        for env in envs:
            x, y = batch[env]

            with torch.enable_grad():
                logits = self.model(x)
            
            outputs[env] = {
                "logits": logits,
                "targets": y,
                "penalty": self.compute_penalty(logits, y)
            }

        return outputs
    

    def validation_step_end(self, outputs):
        loss = 0
        preds = []
        ys = []
        for env, output in outputs.items():
            logits = output["logits"]
            y = output["targets"]
            penalty = output["penalty"]

            env_loss = self.loss(logits, y)
            loss += env_loss + self.lamb*penalty
            
            ys.append(y)
            preds.append(argmax(logits, dim=1))
        y = torch.cat(ys, dim=0)
        preds = torch.cat(preds, dim=0)

        # Log PA in the last batch of the epoch. Log every n epochs.
        # if self.trainer.is_last_batch and self.current_epoch % 2 == 0:
        #     self.PA.update(deepcopy(self.model))
        #     pa_dict = self.PA.compute()
        #     metrics_dict = {
        #         "val/logPA": pa_dict["logPA"],
        #         "val/beta": pa_dict["beta"],
        #         "val/PA": pa_dict["PA"],
        #         "val/AFR pred": pa_dict["AFR pred"],
        #         "val/AFR true": pa_dict["AFR true"],
        #         "val/acc_pa": pa_dict["acc_pa"],
        #     }
        #     self.log_dict(metrics_dict, prog_bar=False, on_step=False, on_epoch=True, logger=True, sync_dist=True)

        # Log validation metrics
        metrics_dict = {
            "val/loss": loss,
            "val/acc": self.val_acc(preds, y),
            "val/f1": self.val_f1(preds, y),
            "val/sensitivity": self.val_sensitivity(preds, y),
            "val/specificity": self.val_specificity(preds, y),
            "val/precision": self.val_precision(preds, y),
        }
        self.log_dict(metrics_dict, prog_bar=False, on_step=True, on_epoch=True, logger=True, sync_dist=True)

        # Return loss for scheduler
        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        if (self.hparams.scheduler is not None) and self.trainer.datamodule.val_dataloader():
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        
        return {"optimizer": optimizer}
    

class IRMMnist(IRM):
    def __init__(
        self,
        n_classes: int,
        net: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler,
        lamb: float = 1.0  # Penalty weight for IRM
    ):
        super().__init__(n_classes, net, optimizer, scheduler, lamb)

        self.model = net
        self.loss = nn.CrossEntropyLoss()

class IRMPerceptron(IRM):
    def __init__(self, lr, weight_decay, n_classes, optimizer, momentum, lamb: float = 1.0):
        super().__init__(lr, weight_decay, n_classes, optimizer, momentum, lamb)

        self.model = nn.Linear(2, 2)
        self.n_classes = n_classes

