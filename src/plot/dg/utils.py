from collections import defaultdict
import os
import os.path as osp
from tqdm import tqdm
import wandb

import pandas as pd


def create_dataframe_from_wandb_runs(
    project: str,
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
    date_start (str): a date in the normal date format. All runs after that
        date will be retrieved.
    date_end (str): a date in the normal date format. All runs before that
        date will be retrieved.
    asr (str): string specifying the ASR metric to be used. Can be one of
        "true" or "clean". Default: "true"
    cache (bool): boolean specifying whether to store and retrieve the
        DataFrame for subsequent uses, instead of recomputing it from scratch.
        Default: False.
    """
    if afr not in ("true", "pred"):
        raise ValueError(
            f"'asr' must be one of 'true' or 'pred'. {afr} received instead."
        )

    api = wandb.Api(timeout=100)
    runs = api.runs(project, filters)

    dirname = osp.join("results", "dataframes")
    fname = osp.join(dirname, f"diagvib_afr={afr}_{date}.pkl")
    os.makedirs(dirname, exist_ok=True)

    if cache and osp.exists(fname):
        return pd.read_pickle(fname)

    data = defaultdict(list)

    for i, run in tqdm(enumerate(runs), total=len(runs)):
        # if "sr" not in run.name[-9:]:
        #     continue
        # if not run.config["max_beta/exp_name"].endswith("robust2"):
        #     continue

        # if not len(run.history()):
        #     print(run.name)
        #     continue

        config = run.config
        history = run.history()

        data["name"].append(run.name)
        data["shift_ratio"].append(config["data/dg/shift_ratio"])
        data["model_name"].append(config["model/dg/classifier/exp_name"])
        data["shift_factor"].append(config["data/dg/ds2_env"])
        data["AFR"].append(history[f"AFR {afr}"].max())
        data["logPA"].append(history["logPA_epoch"].max())
        data["beta"].append(history["beta_epoch"][history["logPA_epoch"].argmax()])

    df = pd.DataFrame(data)
    df.to_pickle(fname)
    print(f"dataframe stored in {fname}.")

    return df
