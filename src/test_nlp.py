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
    """Tests the model using Posterior Agreement on different data shifts:
        - shift_factor: Associated with the distribution shift between two datasets to compare.
        - shift_ratio: Associated with the percentage of the shifted data in the second dataset.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    # set seed for random number generators in pytorch, numpy and python.random
    pl.seed_everything(cfg.seed, workers=True)

    # We load the experiment configuration: experiment_name and seed
    path_ckpt_csv = r"/cluster/home/vjimenez/adv_pa_new/logs/victor/NLP.csv"
    experiment_df = pd.read_csv(path_ckpt_csv)

    if cfg.logger.wandb is not None:
        group = cfg.logger.wandb.group
    else:
        group = cfg.group

    selected_ckpt = experiment_df[
        (experiment_df['group'] == group) & (experiment_df['experiment_name'] == cfg.experiment_name) & (experiment_df['seed'] == str(cfg.seed)) & (experiment_df['metric'] == str(cfg.checkpoint_metric))
    ]
    assert len(selected_ckpt) == 1, "There are duplicate experiments in the csv file."
    ckpt_path = selected_ckpt["ckpt_path"].item()

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
        model = model.load_from_checkpoint(ckpt_path),
        datamodule=datamodule,
    )

    test_metrics = trainer.callback_metrics
    metric_dict = {**test_metrics}
    return metric_dict, object_dict


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="test_datashift.yaml"
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
