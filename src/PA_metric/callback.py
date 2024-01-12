from typing import Optional, List
import torch.nn.functional as F

# To implement it in a LightningModule without receiving errors
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import Trainer, LightningModule, LightningDataModule
from copy import deepcopy

from .metric import PosteriorAgreement

class PA_Callback(Callback):
    def __init__(self,
                log_every_n_epochs: int,
                pa_epochs: int,
                datamodule: LightningDataModule,
                beta0: Optional[float],
                early_stopping: Optional[List] = None):
        
        """
        Incorporation of the PA Metric to the Lightning training procedure. A LightningDataModule is required to generate the logits, and this
        is either provided during initialization or else the validation DataLoader is used.
        """
        super().__init__()

        self.beta0 = beta0
        self.pa_epochs = pa_epochs
        self.log_every_n_epochs = log_every_n_epochs
        self.early_stopping = early_stopping

        # TODO: Check that the dataset is not stored twice if its the same as the validation dataset.
        datamodule.prepare_data()
        datamodule.setup()
        self.PosteriorAgreement = PosteriorAgreement(
                                    dataset = datamodule.test_pairedds,
                                    beta0 = self.beta0,
                                    pa_epochs = self.pa_epochs,
                                    early_stopping = self.early_stopping,
                                    strategy = "lightning"
                                    )

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        if (pl_module.current_epoch + 1) % self.log_every_n_epochs == 0:
            self.PosteriorAgreement.update(classifier=deepcopy(pl_module.model))
            pa_dict = self.PosteriorAgreement.compute()
            metrics_dict = {
                "val/logPA": pa_dict["logPA"],
                "val/beta": pa_dict["beta"],
                "val/PA": pa_dict["PA"],
                "val/AFR pred": pa_dict["AFR pred"],
                "val/AFR true": pa_dict["AFR true"],
                "val/acc_pa": pa_dict["acc_pa"],
            }
            self.log_dict(metrics_dict, prog_bar=False, on_step=False, on_epoch=True, logger=True, sync_dist=True)