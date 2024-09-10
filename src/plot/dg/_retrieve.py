import os
import json
import os.path as osp
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt

import wandb
import pickle

from src.plot.dg import *

def retrieve_from_history(run, keyname):
    return [row[keyname] for row in run.scan_history(keys=[keyname]) if row[keyname] is not None]

def parse_wandb_table(df):
    df_dict = json.loads(",".join(df.columns.to_list()).replace('}.1,', '},'))
    return np.asarray(df_dict["data"])

def get_table_name(artifact_name):
    return artifact_name.split('-')[-1].split(':')[0]

def dg_pa_datashift(
        project: str,
        group: str,
        run_name: str,
        filters: dict,
        labelwise: bool = False,
        dirname: str = "results",
        cache: bool = False,
    ) -> pd.DataFrame:

    api = wandb.Api(timeout=100)
    runs = api.runs(project, filters)

    pathdir = osp.join(dirname, group)
    fname = osp.join(pathdir, f"{run_name}.pkl")
    os.makedirs(dirname, exist_ok=True)

    if cache and osp.exists(fname):
        with open(fname, 'rb') as file:
            loaded_dict = pickle.load(file) 
        return loaded_dict
    
    data = defaultdict(list)
    for i, run in tqdm(enumerate(runs), total=len(runs)):
        config = run.config        

        if run.name != run_name:
            continue
        
        # Config keys
        data["seed"].append(config["seed"])
        data["dataset"].append(group)
        data["model"].append(config["model/_target_"].split(".")[-1])
        try:
            data["trainer"].append(config['trainer/strategy'])
        except:
            data["trainer"].append("gpu")
        data["epochs"].append(config['trainer/max_epochs'])
        data["n_classes"].append(config["n_classes"])
        data["batch_size"].append(config['data/batch_size'])
        data["net"].append(config['model/net/net'])
        data["pretrained"].append(config['model/net/pretrained'])
        data["loss"].append(config['model/loss/_target_'].split(".")[-1])
        data["optimizer"].append(config["optimizer"])
        data["scheduler"].append(config["scheduler"])

        # Metrics:
        data["MSE"].append(
                    np.asarray(retrieve_from_history(run, "PA/MSE"))
        )
        for env in range(1,6):
            for metric in LIST_PASPECIFIC_METRICS:
                metric_key = f"PA(0,{env})/{metric}"
                data[metric_key].append(
                    np.asarray(retrieve_from_history(run, metric_key))
                )

            if labelwise:
                for lab in range(int(config["n_classes"])):
                    data[metric_key + f"@{lab}"].append(retrieve_from_history(run, metric_key + f"_{lab}"))

        # Now we get training metrics:
        for stage in ["train", "val"]:
            for metric_stage in ["loss", "acc", "specificity", "sensitivity", "precision"]:
                try:
                    data[f"{stage}/{metric_stage}"].append(
                            np.asarray(retrieve_from_history(run, f"{stage}/{metric_stage}_epoch"))
                        )
                except:
                    import ipdb; ipdb.set_trace()

        # Finally the test/oracle metrics:
        for i in range(6):
            data[f"oracle/acc_{str(i)}"].append(
                    np.asarray(retrieve_from_history(run, f"oracle/acc_{str(i)}_epoch"))
            )

        # Now we get the tables:
        all_dfs = {} # Dictionary to store all DataFrames
        artifacts = run.logged_artifacts()
        for artifact in artifacts:
            try:
                # Download the artifact
                artifact_dir = artifact.download()
                files = os.listdir(artifact_dir)
                if files:
                    all_dfs[artifact.name] = pd.read_csv(
                        os.path.join(artifact_dir, files[0])
                    )
                    # print(f"Successfully retrieved {artifact.name}")
                else:
                    print(f"No files found in the artifact directory for {artifact.name}")
            except Exception as e:
                print(f"Failed to retrieve {artifact.name}: {str(e)}")

        tables_dict = {}
        for name, df in all_dfs.items():
            try:
                tables_dict[get_table_name(name)] = parse_wandb_table(df)
            except Exception as e:
                print(f"Failed to parse {name}: {str(e)}")

        for table_name in tables_dict.keys():
            if table_name[3].isdigit() == False: 
                data[table_name] = tables_dict[table_name]

            if (table_name[3].isdigit() == True) and (labelwise == True):
                data[table_name] = tables_dict[table_name]

        # Store it already, only one dictionary per run:
        if not osp.exists(pathdir):
            os.mkdir(pathdir)
        with open(fname, 'wb') as file:
            pickle.dump(data, file)
    
        return data


def get_dictionary(dataset_name, run_names: list, datashift: bool = True):
    entity = "malvai"
    project = 'DiagVib-6 Paper' if datashift==True else 'DiagVib-6 OOD Model Selection'
    data_dict_list = []
    folder = "datashift" if datashift==True else "modelselection"
    for run_name in tqdm(run_names, desc="Run: "):
        data_dict = dg_pa_datashift(
            project = entity + '/' + project,
            group = dataset_name,
            run_name = run_name,
            filters= {
                    "group": dataset_name,
            },
            labelwise = False,
            cache = True,
            dirname = fr"/cluster/home/vjimenez/adv_pa_new/results/dg/{folder}"
        )
        data_dict_list.append(data_dict)

    merged_dict = defaultdict(list)
    for d in data_dict_list:
        for key, value in d.items():
            merged_dict[key].extend(value)

    return merged_dict

def get_multiple_dict(dataset_names, run_names, datashift: bool = True):
    data_dict_list_2 = []
    for dataset_name in dataset_names:
        data_dict_list_2.append(
            get_dictionary(dataset_name, run_names, datashift)
        )

    merged_dict = defaultdict(list)
    for d in data_dict_list_2:
        for key, value in d.items():
            merged_dict[key].extend(value)

    return merged_dict