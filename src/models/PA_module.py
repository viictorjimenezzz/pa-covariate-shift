from typing import Optional, List, Any

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import pytorch_lightning as pl

from src.models.components.post_agr import PosteriorAgreementKernel
from torchmetrics.classification.accuracy import Accuracy


class PosteriorAgreementModule(pl.LightningModule):
    """Optimization over the inverse temperature parameter of the Posterior Agreement kernel.
    """

    def __init__(
        self,
        classifier: nn.Module,
        optimizer: torch.optim.Optimizer,
        beta0: float,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        for param in classifier.parameters():
            param.requires_grad = False

        self.model = classifier.eval()
        self.kernel = PosteriorAgreementKernel(beta0=beta0)
        self.afr_true = Accuracy(task="multiclass", num_classes=10) # FIX THIS LATER TOO
        self.afr_pred = Accuracy(task="multiclass", num_classes=10)

        # self.betas = np.linspace(0, 10, 100).tolist()
        # self.training_step_outputs_1 = []
        # self.training_step_outputs_2 = []

    def model_step(self, batch: dict):
        # QUICK FIX for projection. TODO: implement a better version

        env_names = batch["envs"]
        self.kernel.beta.data.clamp_(min=0.0)
        self.kernel.reset()
        x1, x2 = batch[env_names[0]][0], batch[env_names[1]][0]

        with torch.no_grad():
            o1, o2 = self.model(x1), self.model(x2)

        self.kernel(o1, o2)

        loss = -self.kernel.log_posterior()
        return o1, o2, loss

    def training_step(self, train_batch: Any, batch_idx: int):
        o1, o2, loss = self.model_step(train_batch)

        if self.current_epoch == 0:  # AFR does not change during the epochs
            y_pred = torch.argmax(o1.data, 1)
            y_pred_adv = torch.argmax(o2.data, 1)
            y_true = train_batch["0"][1]

            self.afr_pred(y_pred_adv, y_pred)
            self.afr_true(y_pred_adv, y_true)

            self.log(
                "AFR pred",
                self.afr_pred,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

            self.log(
                "AFR true",
                self.afr_true,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        # self.training_step_outputs_1.append(o1)
        # self.training_step_outputs_2.append(o2)
        self.kernel(o1, o2)

        values = {
            "beta": self.kernel.beta.item(),
            "logPA": self.kernel.log_posterior().item(),
            "PA": self.kernel.posterior(),
        }

        self.log_dict(
            values,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
        )

        return loss

    # def on_train_epoch_end(self):
    #     if self.current_epoch == 0:
    #         PAs = []
    #         log_PAs = []
    #         for beta in self.betas:
    #             kernel = PosteriorAgreementKernel(beta0=beta)
    #             kernel.to(self.kernel.beta.device)
    #             kernel.eval()
    #             with torch.no_grad():
    #                 for o1, o2 in zip(
    #                     self.training_step_outputs_1,
    #                     self.training_step_outputs_2,
    #                 ):
    #                     o1.to(kernel.beta.device)
    #                     o2.to(kernel.beta.device)
    #                     kernel(o1, o2)
    #                 PAs.append(kernel.posterior().cpu().numpy())
    #                 log_PAs.append(kernel.log_posterior().cpu().numpy())

    #         values = {
    #             "all_betas": self.betas,
    #             "all_logPA": PAs,
    #             "all_PA": log_PAs,
    #         }
    #         self.log_dict(
    #             values,
    #             prog_bar=True,
    #             on_step=True,
    #             on_epoch=True,
    #             logger=True,
    #         )

    #         # fig1, ax1 = plt.subplots()
    #         # ax1.plot(self.betas, PAs)
    #         # ax1.set_xlabel("beta")
    #         # ax1.set_ylabel("PA")

    #         # fig2, ax2 = plt.subplots()
    #         # ax2.plot(self.betas, log_PAs)
    #         # ax2.set_xlabel("Beta")
    #         # ax2.set_ylabel("log_PA")
    #         # logger = self.logger.experiment
    #         # logger.log({"PA_vs_Beta":fig1})
    #         # logger.log({"logPA_vs_Beta": fig2})

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(self.parameters())
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = PosteriorAgreementModule(nn.Identity(), None, None)
