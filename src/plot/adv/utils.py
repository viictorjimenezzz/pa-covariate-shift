from collections import defaultdict

import os
import os.path as osp

from tqdm import tqdm

import pandas as pd

import wandb
import time


def create_dataframe_from_wandb_runs(
    project: str,
    attack: str,
    filters: dict,
    date: str = None,
    afr: str = "true",
    cache: bool = False,
) -> pd.DataFrame:
    """
    Create a pandas DataFrame from the data retrieved by Weight & Biases. The
    data can be used for plotting.

    project (str): name of the W&B project (e.g. <entity>/<project>).
    dset (str): name of the dataset (currently used only for the fname if
        cache=True).
    attack (str): name of the attack. Can be one of "PGD" or "FMN"
    filters (dict): a dictionary containing conditions to filter out runs. The
        syntax is that defined by W&B, based on MongoDB.
    date (str): a date in the normal date format. All runs after that date will
        be retrieved.
    afr (str): string specifying the AFR metric to be used. Can be one of
        "true" or "pred". Default: "true"
    cache (bool): boolean specifying whether to store and retrieve the
        DataFrame for subsequent uses, instead of recomputing it from scratch.
        Default: False.
    """
    if afr not in ("true", "pred"):
        raise ValueError(
            f"'afr' must be one of 'true' or 'pred'. {afr} received instead."
        )

    api = wandb.Api(timeout=100)
    runs = api.runs(project, filters)

    dirname = osp.join("results", "dataframes")
    fname = osp.join(dirname, f"cifar10_{attack}_afr={afr}_{date}.pkl")
    os.makedirs(dirname, exist_ok=True)

    if cache and osp.exists(fname):
        return pd.read_pickle(fname)

    data = defaultdict(list)

    for run in tqdm(runs, total=len(runs)):
        config = run.config
        history = run.history()

        data["name"].append(run.name)
        data["attack_name"].append(
            config.get(
                "data/attack/attack_name",
                config.get("data/adv/attack/attack_name"),
            )
        )
        data["model_name"].append(
            config.get(
                "data/classifier/model_name",
                config.get("data/adv/classifier/model_name"),
            )
        )
        data["adversarial_ratio"].append(
            config.get(
                "data/adversarial_ratio",
                config.get("data/adv/adversarial_ratio"),
            )
        )

        if "data/attack/epsilons" in config or "data/adv/attack/epsilons" in config:
            data["linf"].append(
                config.get(
                    "data/attack/epsilons",
                    config.get("data/adv/attack/epsilons"),
                )
            )
        # pause because wandb sometimes is not able to retrieve the results
        time.sleep(3)
        data["AFR"].append(
            max(
                [
                    row["AFR pred"]
                    for row in run.scan_history()
                    if row["AFR pred"] is not None
                ]
            )
        )
        data["logPA"].append(history["logPA_epoch"].max())

    df = pd.DataFrame(data)
    df.to_pickle(fname)
    print(f"dataframe stored in {fname}.")

    return df
