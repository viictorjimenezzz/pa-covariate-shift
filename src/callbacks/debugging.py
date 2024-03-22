from pytorch_lightning.callbacks import Callback
from torchmetrics import Accuracy
import torch

class Debugging_Callback(Callback):
    """
    Callback used to accumulate and print metrics along the training pipeline for debugging purposes.
    """
    def __init__(self):
        super().__init__()

        self.len_train = torch.tensor(0)
        self.right_train = torch.tensor(0)

        self.len_val = torch.tensor(0)
        self.right_val = torch.tensor(0)

        self.len_test = torch.tensor(0)
        self.right_test = torch.tensor(0)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        y, preds = outputs["targets"], outputs["preds"]

        # Update the metrics
        self.len_train = torch.tensor([self.len_train.item() + len(y)]).to(pl_module.device)
        self.right_train = torch.tensor([self.right_train.item() + torch.sum(torch.eq(y, preds)).item()]).to(pl_module.device)

        metrics_dict = {
            'debug/train_len': self.len_train,
            'debug/train_right': self.right_train
        }
        pl_module.log_dict(metrics_dict, prog_bar=False, on_step=False, on_epoch=True, logger=True, sync_dist=True)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        y, preds = outputs["targets"], outputs["preds"]

        # Update the metrics
        self.len_val = torch.tensor([self.len_val.item() + len(y)]).to(pl_module.device)
        self.right_val = torch.tensor([self.right_val.item() + torch.sum(torch.eq(y, preds)).item()]).to(pl_module.device)

        metrics_dict = {
            'debug/val_len': self.len_val,
            'debug/val_right': self.right_val
        }
    
        pl_module.log_dict(metrics_dict, prog_bar=False, on_step=False, on_epoch=True, logger=True, sync_dist=True)

    def on_fit_end(self, trainer, pl_module):
        print("\nTraining is concluded, showing debugging metrics:")
        print("Length train, val: ", self.len_train.item(), self.len_val.item())
        print("Accuracy train, val: ", self.right_train.item() / self.len_train.item(), self.right_val.item() / self.len_val.item())

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        y, preds = outputs["targets"], outputs["preds"]

        # Update the metrics
        self.len_test = torch.tensor([self.len_test.item() + len(y)]).to(pl_module.device)
        self.right_test = torch.tensor([self.right_test.item() + torch.sum(torch.eq(y, preds)).item()]).to(pl_module.device)

        metrics_dict = {
            'debug/train_len': self.len_test,
            'debug/train_right': self.right_test
        }
       
        pl_module.log_dict(metrics_dict, prog_bar=False, on_step=False, on_epoch=True, logger=True, sync_dist=False) # SINGLE DEVICE

    def on_test_end(self, trainer, pl_module):
        print("\nTesting is is concluded, showing debugging metrics:")
        print("Length test: ", self.len_test.item())
        print("Accuracy test: ", self.right_test.item() / self.len_test.item())