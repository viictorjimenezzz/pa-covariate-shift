from pytorch_lightning import LightningModule
from torch import nn, argmax, optim
from torchmetrics import Accuracy, F1Score, Recall, Specificity, Precision


class ERM(LightningModule):
    """Vanilla ERM traning scheme for fitting a NN to the training data"""

    def __init__(
        self,
        n_classes: int,
        net: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler,
    ):
        super().__init__()

        self.model = None
        self.loss = None

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

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)

        return {"logits": logits, "targets": y}

    def training_step_end(self, outputs):
        logits = outputs["logits"]
        y = outputs["targets"]

        loss = self.loss(input=logits, target=y)
        preds = argmax(logits, dim=1)

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
        x, y = batch
        logits = self.model(x)

        return {"logits": logits, "targets": y}

    def validation_step_end(self, outputs):
        logits = outputs["logits"]
        y = outputs["targets"]

        loss = self.loss(input=logits, target=y)
        preds = argmax(logits, dim=1)

        self.log("val/loss", loss, prog_bar=False, on_step=True, on_epoch=True)

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
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
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


class ERMPerceptron(ERM):
    def __init__(self, lr, weight_decay, n_classes, optimizer, momentum):
        super().__init__(lr, weight_decay, n_classes, optimizer, momentum)

        self.model = nn.Linear(2, 2)
        self.n_classes = n_classes


class ERMMnist(ERM):
    def __init__(
        self,
        n_classes: int,
        net: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler,
    ):
        super().__init__(n_classes, net, optimizer, scheduler)

        self.model = net
        self.loss = nn.CrossEntropyLoss()
