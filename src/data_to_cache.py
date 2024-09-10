from typing import List, Optional, Tuple

import hydra
import pytorch_lightning as pl
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import (
    LightningDataModule,
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
def data_to_cache(cfg: DictConfig) -> Tuple[dict, dict]:
    """
    Calls the .setup() method of the datamodule to cache data files. It is intended to be executed in a single process.
    """

    # set seed for random number generators in pytorch, numpy and python.random
    pl.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup("fit")
    datamodule.setup("test")
    return


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="train.yaml"
)
def main(cfg: DictConfig) -> Optional[float]:
    # train the model
    data_to_cache(cfg)

    return


if __name__ == "__main__":
    main()
