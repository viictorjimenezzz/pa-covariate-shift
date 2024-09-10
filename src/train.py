from typing import List, Optional, Tuple

import hydra
import os
import pandas as pd
import csv
import torch.distributed as dist
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
    
    if 'resume_training' in cfg.auxiliary_args.keys():
        if cfg.auxiliary_args.resume_training == True:
            csv_name = "ckpt_exp"
            if 'wandb' in cfg.logger.keys():
                csv_name = cfg.logger.wandb.project

            path_ckpt_csv = cfg.paths.log_dir + f"/{csv_name}.csv"
            experiment_df = pd.read_csv(path_ckpt_csv)
            selected_ckpt = experiment_df[
                (experiment_df['experiment_name'] == cfg.logger.wandb.name) & (experiment_df['metric'] == "last") & (experiment_df['seed'] == str(cfg.seed))
            ]
            assert len(selected_ckpt) == 1, "There are duplicate experiments in the csv file."
            ckpt_path = selected_ckpt["ckpt_path"].item()
            model = model.load_from_checkpoint(
                    ckpt_path,
                    net=hydra.utils.instantiate(cfg.model.net),
                    loss=hydra.utils.instantiate(cfg.model.loss)
            )
    
    # Train model
    log.info("Starting training!")
    trainer.fit(
        model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path")
    )

    # ----------------------------------------------------------------------------------------------
    # DELETE ONCE THE DATA HAS BEEN STORED FOR ALL CASES.
    # import pickle
    # import os

    # RESULTS_DIR = r"/cluster/home/vjimenez/adv_pa_new/results/adv"
    # # fname = f"PGD_eps={cfg.auxiliary_args.epsilons}_ar=1.0_distributions.pkl"
    # fname = f"FMN_ar=1.0_distributions.pkl"
    # file_path = os.path.join(RESULTS_DIR, cfg.model.net.model_name, fname)

    # orig_true_mean = model.stored_orig_true/model.num_orig_true
    # gibbs_orig_true_mean = model.stored_orig_gibbs_true/model.num_orig_true

    # orig_false_mean = model.stored_orig_false/model.num_orig_false
    # gibbs_orig_false_mean = model.stored_orig_gibbs_false/model.num_orig_false

    # adv_true_mean = model.stored_adv_true/model.num_adv_true
    # gibbs_adv_true_mean = model.stored_adv_gibbs_true/model.num_adv_true

    # results = {
    #     'orig_true_mean': orig_true_mean,
    #     'orig_true_std': torch.sqrt(model.stored_orig_true_2/model.num_orig_true - orig_true_mean**2),
    #     'gibbs_orig_true_mean': gibbs_orig_true_mean,
    #     'gibbs_orig_true_std': torch.sqrt(model.stored_orig_gibbs_true_2/model.num_orig_true - gibbs_orig_true_mean**2),

    #     'orig_false_mean': orig_false_mean,
    #     'orig_false_std': torch.sqrt(model.stored_orig_false_2/model.num_orig_false - orig_false_mean**2),
    #     'gibbs_orig_false_mean': gibbs_orig_false_mean,
    #     'gibbs_orig_false_std': torch.sqrt(model.stored_orig_gibbs_false_2/model.num_orig_false - gibbs_orig_false_mean**2),
        

    #     'adv_true_mean': adv_true_mean,
    #     'adv_true_std': torch.sqrt(model.stored_adv_true_2/model.num_adv_true - adv_true_mean**2),
    #     'gibbs_adv_true_mean': gibbs_adv_true_mean,
    #     'gibbs_adv_true_std': torch.sqrt(model.stored_adv_gibbs_true_2/model.num_adv_true - gibbs_adv_true_mean**2)
    # }
    # with open(file_path, 'wb') as f:
    #     pickle.dump(results, f)

    # raise NotImplementedError
    # ----------------------------------------------------------------------------------------------


    train_metrics = trainer.callback_metrics
    metric_dict = {**train_metrics}

    # Store model checkpoint path
    csv_name = "ckpt_exp"
    if 'wandb' in cfg.logger.keys() and cfg.logger['wandb'] != None:
        csv_name = cfg.logger.wandb.project
    path_ckpt_csv = cfg.paths.log_dir + f"/{csv_name}.csv"

    # On rank 0, we log the experiment.
    is_rank_0 = dist.get_rank() == 0 if dist.is_initialized() else True
    if is_rank_0 and trainer.checkpoint_callbacks[0].monitor != None:
        # Multiple checkpointing callbacks possible:
        ckpt_paths = [ckpt_callb.best_model_path for ckpt_callb in trainer.checkpoint_callbacks]
        tracked_metric_ckpt = [ckpt_callb.monitor.split("/")[-1] for ckpt_callb in trainer.checkpoint_callbacks]
        if os.path.exists(path_ckpt_csv) == False:
            with open(path_ckpt_csv, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["group", "experiment_name", "experiment_id", "seed", "metric", "ckpt_path"])
                writer.writerow(["place_holder", "place_holder", "place_holder", "place_holder", "place_holder", "place_holder"])

        pd_ckpt = pd.read_csv(path_ckpt_csv)
        if logger:
            with open(path_ckpt_csv, "a+", newline="") as file:
                writer = csv.writer(file)
                for metric, ckpt_path in zip(tracked_metric_ckpt, ckpt_paths):
                    writer.writerow([logger[0].experiment.group, logger[0].experiment.name, logger[0].experiment.id, cfg.seed, metric, ckpt_path])
            
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
