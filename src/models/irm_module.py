from pytorch_lightning import LightningModule
import torch
from torch import nn, argmax, optim
from torch.autograd import grad
from torchmetrics import Accuracy, F1Score, Recall, Specificity, Precision



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

        self.save_hyperparameters(
            ignore=["net"]
        )  # for easier retrieval from w&b and sanity checks

        # Metrics
        self.n_classes = int(n_classes)

        _task = "multiclass" if self.n_classes > 2 else "binary"
        self.train_acc = Accuracy(
            task=_task, num_classes=self.n_classes, average="macro"
        )
        self.train_f1 = F1Score(
            task=_task, num_classes=self.n_classes, average="macro"
        )

        self.train_specificity = Specificity(
            task=_task, num_classes=self.n_classes, average="macro"
        )

        self.train_sensitivity = Recall(
            task=_task, num_classes=self.n_classes, average="macro"
        )
        self.train_precision = Precision(
            task=_task, num_classes=self.n_classes, average="macro"
        )

        self.val_acc = Accuracy(
            task=_task, num_classes=self.n_classes, average="macro"
        )
        self.val_f1 = F1Score(
            task=_task, num_classes=self.n_classes, average="macro"
        )
        self.val_specificity = Specificity(
            task=_task, num_classes=self.n_classes, average="macro"
        )
        self.val_sensitivity = Recall(
            task=_task, num_classes=self.n_classes, average="macro"
        )
        self.val_precision = Precision(
            task=_task, num_classes=self.n_classes, average="macro"
        )

    def compute_penalty(self, logits, y):
        dummy_w = torch.tensor(1.).to(self.device).requires_grad_()
        with torch.enable_grad():
            loss = self.loss(logits*dummy_w, y)
        gradient = grad(loss, [dummy_w], create_graph=True)[0]
        return torch.sum(gradient**2).to(self.device)

    def training_step(self, batch, batch_idx):
        envs = batch["envs"]

        outputs = {} # I try to conserve structure of ERM module
        for env in envs:
            x, y = batch[env]
            self.model.train() # to delete afterwards
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

        self.log(
            "train/loss",
            loss,
            prog_bar=False,
            on_step=True,
            on_epoch=True,
        )

        self.train_acc(preds, y)
        self.log(
            "train/acc",
            self.train_acc,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.train_f1(preds, y)
        self.log(
            "train/f1",
            self.train_f1,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.train_sensitivity(preds, y)
        self.log(
            "train/sensitivity",
            self.train_sensitivity,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.train_specificity(preds, y)
        self.log(
            "train/specificity",
            self.train_specificity,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.train_precision(preds, y)
        self.log(
            "train/precision",
            self.train_precision,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

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

        self.log(
            "val/loss", 
            loss, 
            prog_bar=False, 
            on_step=True, 
            on_epoch=True
        )

        self.val_acc(preds, y)
        self.log(
            "val/acc",
            self.val_acc,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.val_f1(preds, y)
        self.log(
            "val/f1",
            self.val_f1,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.val_sensitivity(preds, y)
        self.log(
            "val/sensitivity",
            self.val_sensitivity,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.val_specificity(preds, y)
        self.log(
            "val/specificity",
            self.val_specificity,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.val_precision(preds, y)
        self.log(
            "val/precision",
            self.val_precision,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        return {"loss": loss}

    def predict_step(self, predic_batch, batch_idx):
        return self.model(predic_batch)

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

