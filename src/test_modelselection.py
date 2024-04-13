from typing import List, Optional, Tuple

import hydra
import pandas as pd
import pytorch_lightning as pl
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
)
from pytorch_lightning.loggers import Logger

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Add resolvers to evaluate operations in the .yaml configuration files
from src.utils.omegaconf import register_resolvers
register_resolvers()

from src import utils
log = utils.get_pylogger(__name__)

@utils.task_wrapper
def test(cfg: DictConfig) -> Tuple[dict, dict]:
    """Tests the model for model selection, by loading the checkpoint associated with a specific experiment, seed 
    and checkpoint_metric.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    """
    - Same `experiment.name`, different `experiment_id` => different `seed`, different `ckpt_path` but track the same metric.
    - Same `experiment.name`, same `experiment_id` => same `seed`, different `ckpt_path` because tracks another metric.
    """

    # set seed for random number generators in pytorch, numpy and python.random
    pl.seed_everything(cfg.seed, workers=True)

    # We load the experiment configuration: experiment_name, seed and metric.
    csv_name = "ckpt_exp"
    if 'wandb' in cfg.logger.keys() and cfg.logger['wandb'] != None:
        csv_name = cfg.logger.wandb.project
    
    path_ckpt_csv = cfg.paths.log_dir + f"/{csv_name}.csv"
    experiment_df = pd.read_csv(path_ckpt_csv)
    group_conf = cfg.logger.wandb.group if 'wandb' in cfg.logger.keys() and cfg.logger['wandb'] != None else "???"
    selected_ckpt = experiment_df[
        (experiment_df['group'] == group_conf) & (experiment_df['experiment_name'] == cfg.name_logger) & (experiment_df['seed'] == str(cfg.seed)) & (experiment_df['metric'] == cfg.checkpoint_metric)
    ]
    assert len(selected_ckpt) == 1, "There are duplicate experiments in the csv file."
    
    ckpt_path = selected_ckpt["ckpt_path"].item()
    if 'wandb' in cfg.logger.keys() and cfg.logger['wandb'] != None:
        cfg.logger.wandb.id = selected_ckpt['experiment_id'].item() # log in the same experiment

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

    # Test model
    log.info("\nStarting test!")
    trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )
    
    # Add metric attribute to the trainer:
    trainer.checkpoint_metric = cfg.checkpoint_metric

    trainer.test(
        model = model.load_from_checkpoint(
            ckpt_path,
            net=hydra.utils.instantiate(cfg.model.net),
            loss=hydra.utils.instantiate(cfg.model.loss)
        ),
        datamodule=datamodule,
    )

    test_metrics = trainer.callback_metrics
    metric_dict = {**test_metrics}
    return metric_dict, object_dict


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="test_modelselection.yaml"
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
