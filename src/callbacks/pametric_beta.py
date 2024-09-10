from typing import List
from pametric.lightning.callbacks.metric import PA_Callback
from pytorch_lightning import Trainer, LightningModule

import pandas as pd
import wandb
import torch

class PA_CallbackBeta(PA_Callback):
    """
    Subclass of the original (pametric) PA_Callback that also logs the beta optimization plot at certain epochs.
    """
    def __init__(self, epochs_to_log_beta: List[int], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epochs_to_log_beta = epochs_to_log_beta

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        super().on_train_epoch_end(trainer=trainer, pl_module=pl_module)
        
        # if (pl_module.current_epoch + 1) % self.log_every_n_epochs == 0:     
        #     pa_dict = self.pa_metric(
        #         classifier=pl_module.model if self.alternative_model is None else self.alternative_model,
        #         local_rank=trainer.local_rank,
        #         # TODO: Consider there's early_stopping in the pl_module. How can I fix that?
        #         destroy_process_group = self.destroy_process_group
        #     )
        #     for env_index, metric_dict in pa_dict.items():
        #         dict_to_log = {
        #             f"PA(0,{env_index+1})/beta": metric_dict["beta"],
        #             f"PA(0,{env_index+1})/logPA": metric_dict["logPA"],
        #             f"PA(0,{env_index+1})/AFR_pred": metric_dict["AFR_pred"],
        #             f"PA(0,{env_index+1})/AFR_true": metric_dict["AFR_true"],
        #             f"PA(0,{env_index+1})/acc_pa": metric_dict["acc_pa"],
        #             "epoch": pl_module.current_epoch
        #         }
        #         pl_module.logger.experiment.log(dict_to_log)

        if pl_module.current_epoch in self.epochs_to_log_beta:
            # Logging betas:
            optimization_dict = {
                "epoch": torch.arange(len(self.pa_metric.betas)),
                "betas": self.pa_metric.betas
            }

            pl_module.logger.experiment.log({
                f"PA/betas_epoch={pl_module.current_epoch}": wandb.plot.line(
                    wandb.Table(dataframe=pd.DataFrame(optimization_dict)), 
                    x="epoch", 
                    y="betas", # y columns
                    title=f"Optimization of beta at epoch {pl_module.current_epoch}",
                )
            })

            # Logging logPAs
            pl_module.logger.experiment.log({
                f"PA/logPAs_epoch={pl_module.current_epoch}": wandb.plot.line_series(
                    xs=[
                        list(range(len(self.pa_metric.logPAs[env_index, :])))
                        for env_index in range(self.pa_metric.len_envmetrics)
                    ],
                    ys=[
                        self.pa_metric.logPAs[env_index, :].tolist()
                        for env_index in range(self.pa_metric.len_envmetrics)
                    ],
                    keys=[
                        f"PA(0,{env_index+1})"
                        for env_index in range(self.pa_metric.len_envmetrics)
                    ],
                    xname = "PA optim. epoch",
                    title=f"Optimized logPA at epoch {pl_module.current_epoch}"
                )
            })


    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        super().on_test_epoch_end(trainer, pl_module)

        # Logging betas:
        optimization_dict = {
            "epoch": torch.arange(len(self.pa_metric.betas)),
            "betas": self.pa_metric.betas
        }
        pl_module.logger.experiment.log({
            f"PA/betas": wandb.plot.line(
                wandb.Table(dataframe=pd.DataFrame(optimization_dict)), 
                x="epoch", 
                y="betas", # y columns
                title=f"Optimization of beta",
            )
        })

        # Logging logPAs
        optimization_dict = {
            "epoch": torch.arange(len(self.pa_metric.logPAs[0, :])),
        }
        optimization_dict.update({
            f"PA(0,{env_index+1})": self.pa_metric.logPAs[env_index, :].tolist()
            for env_index in range(self.pa_metric.len_envmetrics)
        })
        pl_module.logger.experiment.log({
            f"PA/logPAs": wandb.plot.line(
                wandb.Table(dataframe=pd.DataFrame(optimization_dict)), 
                x="epoch", 
                y=[f"PA(0,{env_index+1})" for env_index in range(self.pa_metric.len_envmetrics)], # y columns
                title=f"Optimized logPA",
            )
        })


