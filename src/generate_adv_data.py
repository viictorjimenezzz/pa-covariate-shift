from typing import Optional

import pyrootutils
import hydra
from omegaconf import DictConfig

from pytorch_lightning import seed_everything


pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src import utils


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="generate_adv_data",
)
def main(cfg: DictConfig) -> Optional[float]:
    # from omegaconf import OmegaConf

    # print(OmegaConf.to_yaml(cfg))

    # import torch

    # print(f"Is CUDA available? {torch.cuda.is_available()}")
    # exit()

    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    # create the data
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    dm = hydra.utils.instantiate(cfg.data.adv)
    dm.setup()


if __name__ == "__main__":
    main()
