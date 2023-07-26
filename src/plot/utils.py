from collections import defaultdict

import os
import os.path as osp

from tqdm import tqdm

import pandas as pd

import wandb


def create_dataframe_from_wandb_runs(
    project: str,
    attack: str,
    date: str = None,
    afr: str = "true",
    cache: bool = False,
):
    """
    Create a pandas DataFrame from the data retrieved by Weight & Biases. The
    data can be used for plotting.

    project (str): name of the W&B project (e.g. <entity>/<project>).
    dset (str): name of the dataset (currently used only for the fname if
        cache=True).
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
    filters = {
        "state": "finished",
        "tags": {"$all": ["cifar10", attack]},
    }
    if date is not None:
        filters["created_at"] = {"$gte": date}

    api = wandb.Api(timeout=100)
    runs = api.runs(project, filters)

    dirname = osp.join("results", "dataframes")
    fname = osp.join(dirname, f"cifar10_afr={afr}_{date}.pkl")
    os.makedirs(dirname, exist_ok=True)

    if cache and osp.exists(fname):
        return pd.read_pickle(fname)

    data = defaultdict(list)

    for run in tqdm(runs, total=len(runs)):
        config = run.config
        history = run.history()

        data["name"].append(run.name)
        data["attack_name"].append(config["data/attack/attack_name"])
        data["model_name"].append(config["data/classifier/model_name"])
        data["adversarial_ratio"].append(config["data/adversarial_ratio"])

        if "data/attack/epsilons" in config:
            data["linf"].append(config["data/attack/epsilons"])
        data["AFR"].append(history[f"AFR {afr}"].max())
        data["logPA"].append(history["logPA_epoch"].max())

    df = pd.DataFrame(data)
    df.to_pickle(fname)
    print(f"dataframe stored in {fname}.")

    return df


if __name__ == "__main__":
    create_dataframe_from_wandb_runs(
        project="adv_pa_new",
        date="",
        afr="pred",
        cache=True,
    )
