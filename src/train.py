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

# TODO: Remove after debugging
import torch
import os

@utils.task_wrapper
def train(cfg: DictConfig) -> Tuple[dict, dict]:
    """Trains and optionally evaluates the model.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[str, dict, dict]: Best model path, dict with metrics and dict with all instantiated objects.
    """

    # set seed for random number generators in pytorch, numpy and python.random
    pl.seed_everything(cfg.seed, workers=True)

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(
        cfg.get("callbacks")
    )

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

    # Train model
    log.info("Starting training!")
    trainer.fit(
        model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path")
    )

    train_metrics = trainer.callback_metrics
    metric_dict = {**train_metrics}

    # Store model checkpoint path
    path_ckpt_csv = cfg.paths.log_dir + "/ckpt_exp.csv"

    # Multiple checkpointing callbacks possible:
    ckpt_paths = [ckpt_callb.best_model_path for ckpt_callb in trainer.checkpoint_callbacks]
    tracked_metric_ckpt = [ckpt_callb.monitor.split("/")[-1] for ckpt_callb in trainer.checkpoint_callbacks]
    if os.path.exists(path_ckpt_csv) == False:
        with open(path_ckpt_csv, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["experiment_name", "experiment_id", "seed", "metric", "ckpt_path"])
            writer.writerow(["place_holder", "place_holder", "place_holder", "place_holder", "place_holder"])

    pd_ckpt = pd.read_csv(path_ckpt_csv)
    if logger:
        with open(path_ckpt_csv, "a+", newline="") as file:
            writer = csv.writer(file)
            for metric, ckpt_path in zip(tracked_metric_ckpt, ckpt_paths):
                writer.writerow([logger[0].experiment.name, logger[0].experiment.id, cfg.seed, metric, ckpt_path])
        
    if logger:
        print(f"\nExperiment id: {logger[0].experiment.id}")
    return metric_dict, object_dict


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="train.yaml"
)
def main(cfg: DictConfig) -> Optional[float]:
    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
