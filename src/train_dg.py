from typing import List, Optional, Tuple

import hydra
import os
import pandas as pd
import csv
import pytorch_lightning as pl
from src.models.components.backbone import Backbone
import torch
import torch.nn as nn
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)
import torch.optim as optim
from secml.ml.peval.metrics import CMetricAccuracy
from secml.ml.classifiers import CClassifierPyTorch
from secml.data.loader import CDataLoaderMNIST

from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import Logger

from src.data.components import PairDataset
from src.data.components.adv import AdversarialImageDataset



from src import utils

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def train(cfg: DictConfig) -> Tuple[dict, dict]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.dg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.dg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

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
    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
        
    train_metrics = trainer.callback_metrics
    metric_dict = {**train_metrics}
    

    # Store model checkpoint path
    path_ckpt_csv = cfg.paths.log_dir + '/ckpt_exp.csv'   
    ckpt_path = trainer.checkpoint_callback.best_model_path
    
    if os.path.exists(path_ckpt_csv) == False:
        with open(path_ckpt_csv, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['experiment_name', 'ckpt_path'])
            writer.writerow(['place_holder', 'place_holder'])
    
    pd_ckpt = pd.read_csv(path_ckpt_csv)
    if logger[0].experiment.name not in pd_ckpt['experiment_name'].tolist():
        with open(path_ckpt_csv, 'a+', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([logger[0].experiment.name, ckpt_path])

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="./configs", config_name="train.yaml")
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