from typing import Any, Dict, List, Optional, Tuple
import os
import pandas as pd

import hydra
from omegaconf import DictConfig
import torch
from pytorch_lightning import seed_everything, Callback, LightningModule, LightningDataModule, Trainer
from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.loggers import Logger

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Add resolvers to evaluate operations in the .yaml configuration files
from src.utils.omegaconf import register_resolvers
register_resolvers()

from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)

class AdvModule(LightningModule):
    """Dummy model for testing purposes."""
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        (x, y), (x_adv, _) = batch
        logits = self.model(torch.cat([x, x_adv], dim=0))
        return {
            "loss": torch.tensor(0.0).to(logits.device),
            "logits": logits,
            "targets": torch.cat([y, y], dim=0),
            "preds": torch.argmax(logits, dim=1)
        }

    def configure_optimizers(self):
        return {
            "optimizer": LightningOptimizer(torch.optim.SGD(lr=0.0001, params=self.parameters()))
        }

    def optimizer_step(self, *args, **kwargs):
        for param in self.parameters():
            param.grad = None
        args[-1].__call__()


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    # Model checkpoint paths:
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: torch.nn.Module = hydra.utils.instantiate(cfg.model)

    # Initialize a dummy lightningmodule:
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data, model=model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    # Dummy algorithm for testing:
    algorithm = AdvModule(model)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "algorithm": algorithm,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    log.info("Starting testing!")
    trainer.test(model=algorithm, datamodule=datamodule)

    metric_dict = {**trainer.callback_metrics}
    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()