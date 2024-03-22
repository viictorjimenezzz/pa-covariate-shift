from typing import List, Optional, Tuple

import hydra
import os
import pandas as pd
import csv
import pytorch_lightning as pl
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
)
from pytorch_lightning.loggers import Logger

# Add resolvers to evaluate operations in the .yaml configuration files
OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("len", len)
OmegaConf.register_new_resolver("classname", lambda classpath: classpath.split(".")[-1])

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


from src import utils

log = utils.get_pylogger(__name__)

# TO DELETE AFTER IT'S BEEN DEBUGGED
import torch
import os

@utils.task_wrapper
def test(cfg: DictConfig) -> Tuple[dict, dict]:
    """Tests the model.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    # set seed for random number generators in pytorch, numpy and python.random
    pl.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(
        cfg.get("callbacks")
    )

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    # Test model
    log.info("\nStarting test!")
    trainer = hydra.utils.instantiate(
        cfg.trainer, logger=logger, strategy='gpu', devices = 1
    )

    model: LightningModule = hydra.utils.instantiate(cfg.model)
    trainer.test(
        model = model, # model.load_from_checkpoint(cfg.ckpt_path, net=hydra.utils.instantiate(cfg.model.net)), # because 'net' is not stored in the checkpoint
        datamodule=datamodule,
    )

    print("TESTING IS DONE, SHOW LENGTH TO DEBUG:")
    print("lengths: ", torch.sum(model.len_test))

    test_metrics = trainer.callback_metrics
    metric_dict = {**test_metrics}
    return metric_dict, object_dict


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="test.yaml"
)
def main(cfg: DictConfig) -> Optional[float]:
    # train the model
    metric_dict, _ = test(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
