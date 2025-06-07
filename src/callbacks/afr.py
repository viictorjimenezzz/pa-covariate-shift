from typing import Optional

from pytorch_lightning.callbacks import Callback
from torchmetrics import Accuracy

class AFR_Callback(Callback):
    """
    Computes AFR (T) and AFR (P) metrics for pairs of adversarial predictions.
    """
    def __init__(self, n_classes: int):
        super().__init__()
        self.n_classes = n_classes
        _task = "multiclass" if n_classes > 2 else "binary"

        # Initialize metrics
        self.base = Accuracy(task=_task, num_classes=self.n_classes, average="micro")
        self.afrt = Accuracy(task=_task, num_classes=self.n_classes, average="micro")
        self.afrp = Accuracy(task=_task, num_classes=self.n_classes, average="micro")
      
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx: Optional[int] = 0):
        y, preds = outputs["targets"], outputs["preds"]
        half = preds.shape[0] // 2
        y = y[:half]

        metrics_dict = {
            f"test/base": self.base.to(pl_module.device)(preds[:half], y),
            f"test/afrt": self.afrt.to(pl_module.device)(preds[half:], y),
            f"test/afrp": self.afrp.to(pl_module.device)(preds[:half], preds[half:]),
        }
        pl_module.log_dict(metrics_dict, prog_bar=False, on_step=True, on_epoch=True, logger=True, sync_dist=False) # SINGLE DEVICE