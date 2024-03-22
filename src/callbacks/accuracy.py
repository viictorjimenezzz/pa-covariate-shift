from pytorch_lightning.callbacks import Callback
from torchmetrics import Accuracy, F1Score, Recall, Specificity, Precision

class Accuracy_Callback(Callback):
    """
    Computes and logs general accuracy metrics during training and/or testing.
    """
    def __init__(self, n_classes: int):
        super().__init__()

        self.n_classes = n_classes
        _task = "multiclass" if n_classes > 2 else "binary"

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

        # Test metrics
        self.test_acc = Accuracy(task=_task, num_classes=self.n_classes, average="macro")
        self.test_f1 = F1Score(task=_task, num_classes=self.n_classes, average="macro")
        self.test_specificity = Specificity(task=_task, num_classes=self.n_classes, average="macro")
        self.test_sensitivity = Recall(task=_task, num_classes=self.n_classes, average="macro")
        self.test_precision = Precision(task=_task, num_classes=self.n_classes, average="macro")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        y, preds = outputs["targets"], outputs["preds"]
        
        metrics_dict = {
            "train/loss": outputs["loss"],
            "train/acc": self.train_acc.to(pl_module.device)(preds, y),
            "train/f1": self.train_f1.to(pl_module.device)(preds, y),
            "train/sensitivity": self.train_sensitivity.to(pl_module.device)(preds, y),
            "train/specificity": self.train_specificity.to(pl_module.device)(preds, y),
            "train/precision": self.train_precision.to(pl_module.device)(preds, y),
        }

        pl_module.log_dict(metrics_dict, prog_bar=False, on_step=True, on_epoch=True, logger=True, sync_dist=True)


    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        y, preds = outputs["targets"], outputs["preds"]

        metrics_dict = {
            "val/loss": outputs["loss"],
            "val/acc": self.val_acc.to(pl_module.device)(preds, y),
            "val/f1": self.val_f1.to(pl_module.device)(preds, y),
            "val/sensitivity": self.val_sensitivity.to(pl_module.device)(preds, y),
            "val/specificity": self.val_specificity.to(pl_module.device)(preds, y),
            "val/precision": self.val_precision.to(pl_module.device)(preds, y),
        }
        pl_module.log_dict(metrics_dict, prog_bar=False, on_step=True, on_epoch=True, logger=True, sync_dist=True)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        y, preds = outputs["targets"], outputs["preds"]

        metrics_dict = {
            "test/loss": outputs["loss"],
            "test/acc": self.test_acc.to(pl_module.device)(preds, y),
            "test/f1": self.test_f1.to(pl_module.device)(preds, y),
            "test/sensitivity": self.test_sensitivity.to(pl_module.device)(preds, y),
            "test/specificity": self.test_specificity.to(pl_module.device)(preds, y),
            "test/precision": self.test_precision.to(pl_module.device)(preds, y),
        }
        pl_module.log_dict(metrics_dict, prog_bar=False, on_step=True, on_epoch=True, logger=True, sync_dist=False) # SINGLE DEVICE