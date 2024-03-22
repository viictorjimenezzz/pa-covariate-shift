from collections import defaultdict
import os
import os.path as osp
from tqdm import tqdm
import wandb

import pandas as pd
import numpy as np

import time

def retrieve_from_history(run, keyname):
    return [row[keyname] for row in run.scan_history(keys=[keyname]) if row[keyname] is not None]


def dg_pa_dataframe(
        project: str,
        filters: dict,
        dirname: str,
        afr: str = "pred",
        cache: bool = False,
        ) -> pd.DataFrame:
    
    if afr not in ("true", "pred"):
        raise ValueError(
            f"'asr' must be one of 'true' or 'pred'. {afr} received instead."
        )

    api = wandb.Api(timeout=100)
    runs = api.runs(project, filters)

    fname = osp.join(dirname, f"diagvib_afr={afr}_{filters['$and'][0]['tags']}.pkl")
    os.makedirs(dirname, exist_ok=True)

    if cache and osp.exists(fname):
        return pd.read_pickle(fname)

    data = defaultdict(list)

    for i, run in tqdm(enumerate(runs), total=len(runs)):
        config = run.config

        data["name"].append(run.name)
        data["shift_ratio"].append(config["data/dg/shift_ratio"])
        #data["model_name"].append(config["model/dg/classifier/exp_name"]) # no logits
        data["model_name"].append(config["data/dg/classifier/exp_name"]) # logits
        data["shift_factor"].append(config["data/dg/envs_index"][1]) # for original
        data["AFR_true"].append(max(retrieve_from_history(run, f"val/AFR true")))
        data["AFR_pred"].append(max(retrieve_from_history(run, f"val/AFR pred")))
        data["acc_pa"].append(max(retrieve_from_history(run, "val/acc_pa")))
        logpa_epoch = retrieve_from_history(run, "val/logPA")
        data["logPA"].append(max(logpa_epoch))
        data["beta"].append(retrieve_from_history(run, "val/beta")[np.argmax(logpa_epoch)])
        #data["beta"].append(retrieve_from_history(run, "beta")[np.argmax(logpa_epoch)])

    df = pd.DataFrame(data)
    df.to_pickle(fname)
    print(f"dataframe stored in {fname}.")

    return df
