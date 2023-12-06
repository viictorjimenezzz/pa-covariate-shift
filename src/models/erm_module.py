from pytorch_lightning import LightningModule
from torch import nn, argmax, optim
from torchmetrics import Accuracy, F1Score, Recall, Specificity, Precision

# For the PA metric
from src.pa_metric_torch import PosteriorAgreement
from src.data.diagvib_datamodules import DiagVibDataModuleTestPA
from src.data.components.collate_functions import MultiEnv_collate_fn
from copy import deepcopy

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
        self.save_hyperparameters(ignore=["net"])  # for easier retrieval from w&b and sanity checks

        self.n_classes = int(n_classes)
        _task = "multiclass" if self.n_classes > 2 else "binary"

        # Training metrics
        self.train_acc = Accuracy(task=_task, num_classes=self.n_classes, average="macro")
        self.train_f1 = F1Score(task=_task, num_classes=self.n_classes, average="macro")

        self.train_specificity = Specificity(task=_task, num_classes=self.n_classes, average="macro")
        self.train_sensitivity = Recall(task=_task, num_classes=self.n_classes, average="macro")
        self.train_precision = Precision(task=_task, num_classes=self.n_classes, average="macro")

        self.val_acc = Accuracy(task=_task, num_classes=self.n_classes, average="macro")
        self.val_f1 = F1Score(task=_task, num_classes=self.n_classes, average="macro")
        self.val_specificity = Specificity(task=_task, num_classes=self.n_classes, average="macro")
        self.val_sensitivity = Recall(task=_task, num_classes=self.n_classes, average="macro")
        self.val_precision = Precision(task=_task, num_classes=self.n_classes, average="macro")

        # TO DELETE
        # dm = DiagVibDataModuleTestPA(
        #     envs_index = [0, 1],
        #     shift_ratio = 1.0,
        #     envs_name = "val_repbal", # here the full name not only the environment, as we may want to use test_ or val_ or even a custom name
        #     datasets_dir = "./data/dg/dg_datasets/replicate/",
        #     mnist_preprocessed_path = "./data/dg/mnist_processed.npz",
        #     batch_size = 64,
        #     num_workers = 2,
        #     pin_memory = True,
        #     collate_fn = MultiEnv_collate_fn)
        
        # dm.prepare_data()
        # dm.setup()

        # # DEBUGGING PA
        # self.PA = PosteriorAgreement(
        #             dataset = dm.test_pairedds,
        #             pa_epochs = 100,
        #             early_stopping=[0.001, 5, 10],
        #             strategy = "lightning")

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)

        return {"logits": logits, "targets": y}

    def training_step_end(self, outputs):
        logits, y = outputs["logits"], outputs["targets"]
        preds = argmax(logits, dim=1)

        # Log training metrics
        metrics_dict = {
            "train/loss": self.loss(input=logits, target=y),
            "train/acc": self.train_acc(preds, y),
            "train/f1": self.train_f1(preds, y),
            "train/sensitivity": self.train_sensitivity(preds, y),
            "train/specificity": self.train_specificity(preds, y),
            "train/precision": self.train_precision(preds, y),
        }
        self.log_dict(metrics_dict, prog_bar=False, on_step=True, on_epoch=True, logger=True, sync_dist=True)

        # Return loss for optimization
        return {"loss": metrics_dict["train/loss"]}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)

        return {"logits": logits, "targets": y}

    def validation_step_end(self, outputs):
        logits, y = outputs["logits"], outputs["targets"]
        preds = argmax(logits, dim=1)

        #Log PA in the last batch of the epoch. Log every n epochs.
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
            "val/loss": self.loss(input=logits, target=y),
            "val/acc": self.val_acc(preds, y),
            "val/f1": self.val_f1(preds, y),
            "val/sensitivity": self.val_sensitivity(preds, y),
            "val/specificity": self.val_specificity(preds, y),
            "val/precision": self.val_precision(preds, y),
        }
        self.log_dict(metrics_dict, prog_bar=False, on_step=True, on_epoch=True, logger=True, sync_dist=True)

        # Return loss for scheduler
        return {"loss": metrics_dict["val/loss"]}

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
