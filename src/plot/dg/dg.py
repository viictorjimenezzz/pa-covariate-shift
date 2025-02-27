import os
import os.path as osp
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

import numpy as np

def _check_correlations(subset: pd.DataFrame):
    sr = subset["shift_ratio"].values[0] # all the same
    model_names = subset["model_name"].unique()
    shift_factors = np.sort(subset["shift_factor"].unique())

    def _compute_differences(v1, v2 = None):
        if type(v2) == type(None):
            v2 = v1.copy()
        return np.asarray(v1[:-1]) - np.asarray(v2[1:])

    cor_sf_true_pred = np.zeros(len(shift_factors))
    cor_sf_true_true, cor_sf_pred = cor_sf_true_pred.copy(), cor_sf_true_pred.copy()
    for i in range(len(shift_factors)):
        condition = subset["shift_factor"] == shift_factors[i]
        columns = subset.loc[condition, ["logPA", "AFR_true", "AFR_pred"]]
        cor_sf_true_pred[i] = np.corrcoef(_compute_differences(columns["logPA"]), 
                                     _compute_differences(columns["AFR_true"], columns["AFR_pred"]))[0, 1]
        cor_sf_true_true[i] = np.corrcoef(_compute_differences(columns["logPA"]), 
                                          _compute_differences(columns["AFR_true"]))[0, 1]
        cor_sf_pred[i] = np.corrcoef(_compute_differences(columns["logPA"]), 
                                     _compute_differences(columns["AFR_pred"]))[0, 1]
            
    cor_mod_true = np.zeros(len(model_names))
    cor_mod_pred = np.zeros(len(model_names))
    for i in range(len(model_names)):
        condition = subset["model_name"] == model_names[i]
        columns = subset.loc[condition, ["logPA", "AFR_true", "AFR_pred"]]
        cor_mod_true[i] = np.corrcoef(_compute_differences(columns["logPA"]), 
                                      _compute_differences(columns["AFR_true"]))[0, 1]
        cor_mod_pred[i] = np.corrcoef(_compute_differences(columns["logPA"]), 
                                      _compute_differences(columns["AFR_pred"]))[0, 1]

    return np.mean(cor_sf_true_true), np.mean(cor_sf_true_pred), np.mean(cor_sf_pred), np.mean(cor_mod_true), np.mean(cor_mod_pred)


def logpa(df: pd.DataFrame,
          dirname: str,
          show_acc: bool = True,
          picformat: str = "png",
          metric="logPA"):

    model_names = df["model_name"].unique()
    dirname = osp.join(dirname, "PA" if metric == "logPA" else "AFR")
    os.makedirs(dirname, exist_ok=True)

    # To compute the correlations.
    list_shift_ratios = np.sort(df["shift_ratio"].unique())
    corr_sf_pred = np.zeros(len(list_shift_ratios))
    corr_sf_true_true, corr_sf_true_pred, corr_mod_true, corr_mod_pred = corr_sf_pred.copy(), corr_sf_pred.copy(), corr_sf_pred.copy(), corr_sf_pred.copy()

    pairs = [("shift_ratio", "shift_factor")]
    for levels in tqdm(pairs, total=len(pairs)):
        level, x_level = levels
        if level == "shift_factor":
            continue

        level_name, x_name = (
            ("# Shift Factors", "Shift Ratio")
            if level == "shift_factor"
            else ("Shift Ratio", "# Shift Factors")
        )
        if level == "shift_factor":
            level_set = df[
                df["shift_factor"].str.contains("test_", regex=False)
            ]
            levels = sorted(level_set[level].unique())
        else:
            #level_set = df[df["shift_factor"].str.contains("[0-9]", regex=True)]
            # shift factor always has a value
            level_set = df
            levels = level_set[level].unique()
            level_set.sort_values(x_level, inplace=True)

        fontname = "DejaVu serif"
        font_path = fm.findfont(fm.FontProperties(family=fontname))
        levels = np.concatenate([levels, np.array([levels[0]])])
        for value in tqdm(levels, total=len(levels)):
            subset = level_set.loc[
                level_set[level] == value,
                [
                    "model_name",
                    "shift_ratio",
                    "shift_factor",
                    "logPA",
                    "AFR_true",
                    "AFR_pred",
                    "acc_pa"
                ],
            ].sort_values(by='model_name')
            colors_dict = {model_name: plt.rcParams['axes.prop_cycle'].by_key()['color'][i] for i, model_name in enumerate(model_names)}
            label_dict = {model_name: str(i) for i, model_name in enumerate(model_names)}

            # Addendum for ERM, IRM, LISA
            ids = None
            standard_models = ["Vanilla ERM", "Arjovsky et al.", "Yao et al."]
            if len(model_names) == 3 and [model_name[:3].lower() for model_name in model_names] == ["erm", "irm", "lis"]:
                label_dict = {model_name: standard_models[i] for i, model_name in enumerate(model_names)}
                ids = [0, 1, 2]

            _, ax1 = plt.subplots(figsize=(2 * 3.861, 2 * 2.7291))
            sns.set(font_scale=1.9)
            plt.rcParams["font.family"] = "serif"
            plt.rcParams["font.serif"] = fontname
            sns.set_style("ticks")

            #subset["logPA"] = subset["logPA"]/1000.0
            sns.lineplot(
                data=subset,
                ax=ax1,
                x=x_level,
                y=metric,
                hue="model_name",
                style="model_name",
                palette=colors_dict,
                dashes=False,
                marker="o",
                linewidth=3,
            )

            if show_acc:
                for i, point in subset.iterrows():
                    ax1.text(
                        point['shift_factor'], 
                        point['logPA'],        
                        f"{point['AFR_true']:.2f}", 
                        color='black',       
                        ha='center',       
                        va='bottom',
                        fontsize=9    
                    )

            if level == "shift_ratio":
                ind = int(np.where(list_shift_ratios == value)[0])
                corr_sf_true_true[ind], corr_sf_true_pred[ind], corr_sf_pred[ind], corr_mod_true[ind], corr_mod_pred[ind] = _check_correlations(subset)

            ax1.minorticks_on()
            ax1.tick_params(axis="both", which="both", direction="in")
            xticks_font = fm.FontProperties(family=fontname)

            for tick in ax1.get_xticklabels():
                tick.set_fontproperties(xticks_font)

            ax1.grid(linestyle="--")

            ax1.set_xlabel(x_name, fontname=fontname)
            ax1.set_ylabel("PA", fontname=fontname)
            #ax1.set_ylabel(r"$2 \times 10^3 \cdot$" + " PA", fontname=fontname)
            # ax1.set_ylim(min(subset["logPA"])*2, -5)
            # ax1.set_yscale('symlog')

            # Legend
            handles, labels = ax1.get_legend_handles_labels()
            labels = [label_dict[label] for label in labels]

            # sort labels and handles
            if not ids:
                ids = sorted(range(len(labels)), key=labels.__getitem__)
            labels = [labels[i] for i in ids]
            handles = [handles[i] for i in ids]

            #ax1.legend(handles, labels)
            fontname = "DejaVu Serif"
            _ = fm.findfont(fm.FontProperties(family=fontname))
            ax1.legend(
                handles,
                labels,
                handlelength=0.5,
                prop={"family": fontname, "size": 16},
            )

            if level_name == "Shift Ratio":
                ax1.set_title(f"{level_name} = {value:.4f}", fontname=fontname)
            else:
                ax1.set_title(
                    f"{level_name} = {value.split('test_')[1]}",
                    fontname=fontname,
                )

            plt.tight_layout()
            if level == "shift_ratio":
                fname = osp.join(dirname, f"{list(set(list(subset['model_name'])))[0]}_{level}={value:.3f}." + picformat)
            else:
                fname = osp.join(
                    dirname, f"{level}={value.split('test_')[1]}." + picformat
                )

            plt.savefig(fname)
            plt.clf()
            plt.close()
    
    import ipdb; ipdb.set_trace()


def afr_vs_logpa(df: pd.DataFrame, comparison_metric: str = "ASR"):
    """Create and store plots of Linf/Poison ratio vs comparison_metric/logPA.
    Each plot have Linf/Adversarial Ratio on the x axis and four curves on the y
    axis: comparison_metric for a weak (Standard) and robust model,logPA for a
    weak and robust model.

    df (pandas.DataFrame): the DataFrame generated by create_dataframe
    comparison_metric (str): the metric used for the comparison with logPA
    """
    dirname = osp.join("results", "plots", "dg", "joint")
    os.makedirs(dirname, exist_ok=True)

    pairs = [("shift_factor", "shift_ratio"), ("shift_ratio", "shift_factor")]
    for levels in tqdm(pairs, total=len(pairs)):
        level, x_level = levels
        if level == "shift_factor":
            continue

        level_name, x_name = (
            ("# Shift Factors", "Shift Ratio")
            if level == "shift_factor"
            else ("Shift Ratio", "# Shift Factors")
        )
        if level == "shift_factor":
            level_set = df[
                df["shift_factor"].str.contains("test_", regex=False)
            ]
            levels = sorted(level_set[level].unique())
        else:
            level_set = df[df["shift_factor"].str.contains("[0-9]", regex=True)]
            # level_set = level_set.sort_values(
            #     by=x_level,
            #     key=lambda x: np.argsort(index_natsorted(level_set[x_level]))
            # )
            # levels = sorted(level_set[level].unique())
            levels = level_set[level].unique()

        fontname = "Times New Roman"
        font_path = fm.findfont(fm.FontProperties(family=fontname))
        for value in tqdm(levels, total=len(levels)):
            # Subset the DataFrame to include only the relevant columns and rows
            subset = level_set.loc[
                level_set[level] == value,
                [
                    "model_name",
                    "shift_ratio",
                    "shift_factor",
                    "logPA",
                    comparison_metric,
                ],
            ]

            dashes_dict = {"diagvib_erm": (2, 2), "diagvib_erm_robust": (2, 2)}
            colors_dict = {
                "diagvib_erm": "tab:orange",
                "diagvib_erm_robust": "tab:blue",
            }
            label_dict = {"diagvib_erm": "Weak", "diagvib_erm_robust": "Robust"}

            # if level == "adversarial_ratio":
            #     subset = subset[(subset["linf"] > 0.012) & (subset["linf"] < 0.5)]

            _, ax1 = plt.subplots(
                figsize=(2 * 3.861, 2 * 2.7291),
            )
            sns.set(
                font_scale=1.9,
            )
            plt.rcParams["font.family"] = "serif"
            plt.rcParams["font.serif"] = fontname
            # sns.set_style("whitegrid")
            sns.set_style("ticks")
            # plt.xticks(rotation=30, ha="left")
            # plt.style.use(["science", "grid"])
            ax2 = ax1.twinx()
            # plt.style.use("science")

            sns.lineplot(
                data=subset,
                ax=ax1,
                x=x_level,
                y="logPA",
                hue="model_name",
                style="model_name",
                palette=colors_dict,
                dashes=False,
                marker="o",
                linewidth=3,
            )

            sns.lineplot(
                data=subset,
                ax=ax2,
                x=x_level,
                y="AFR",
                hue="model_name",
                style="model_name",
                palette=colors_dict,
                dashes=dashes_dict,
                marker="X",
                linewidth=3,
            )

            ax1.minorticks_on()
            ax2.minorticks_on()
            ax1.tick_params(axis="both", which="both", direction="in")
            ax2.tick_params(axis="both", which="both", direction="in")
            xticks_font = fm.FontProperties(family=fontname)

            for tick in ax1.get_xticklabels():
                tick.set_fontproperties(xticks_font)

            for tick in ax2.get_xticklabels():
                tick.set_fontproperties(xticks_font)

            ax1.grid(linestyle="--")
            ax2.grid(False)

            ax1.set_xlabel(x_name, fontname=fontname)
            ax1.set_ylabel("LogPA", fontname=fontname)
            # ax1.set_xticklabels(ax1.get_xticks(), rotation=45)
            ax2.set_ylabel(comparison_metric, fontname=fontname)

            # Legend
            handles1, labels1 = ax1.get_legend_handles_labels()
            labels1 = [label_dict[label] + " (logPA)" for label in labels1]
            handles2, labels2 = ax2.get_legend_handles_labels()
            labels2 = [label_dict[label] + " (AFR)" for label in labels2]
            labels = labels1 + labels2
            handles = handles1 + handles2

            # sort labels and handles
            ids = sorted(range(len(labels)), key=labels.__getitem__)
            ids[0], ids[1], ids[2], ids[3] = ids[1], ids[0], ids[3], ids[2]
            labels = [labels[i] for i in ids]
            handles = [handles[i] for i in ids]

            ax2.legend(handles, labels)
            ax1.legend().remove()

            if level_name == "Shift Ratio":
                ax1.set_title(f"{level_name} = {value:.4f}", fontname=fontname)
            else:
                ax1.set_title(
                    f"{level_name} = {value.split('test_')[1]}",
                    fontname=fontname,
                )

            plt.tight_layout()
            if level == "shift_ratio":
                fname = osp.join(dirname, f"{level}={value:.8f}.pdf")
            else:
                fname = osp.join(
                    dirname, f"{level}={value.split('test_')[1]}.pdf"
                )

            plt.savefig(fname)
            plt.clf()
            plt.close()


def afr_vs_logpa_barplots(df: pd.DataFrame, comparison_metric: str = "ASR"):
    dirname = osp.join("results", "plots", "dg", "barplots")
    os.makedirs(dirname, exist_ok=True)

    level, x_level = "shift_ratio", "shift_factor"
    level_name, x_name = "Shift Ratio", "Shift Factor"
    level_set = df[df["shift_factor"].str.contains("test_", regex=False)]
    levels = sorted(level_set[level].unique())

    levels = sorted(level_set[level].unique())
    for metric in ("logPA", comparison_metric):
        for value in tqdm(levels, total=len(levels)):
            # Subset the DataFrame to include only the relevant columns and rows
            subset = level_set.loc[
                level_set[level] == value,
                ["model_name", "shift_ratio", "shift_factor", metric],
            ]

            dashes_dict = {
                "diagvib_erm": (2, 2),
                "diagvib_erm_robust": (2, 2),
            }
            colors_dict = {
                "diagvib_erm": "tab:orange",
                "diagvib_erm_robust": "tab:blue",
            }
            label_dict = {
                "diagvib_erm": "Weak",
                "diagvib_erm_robust": "Robust",
            }

            # sns.set_style("whitegrid")
            # plt.style.use(["science", "grid"])
            # ax2 = ax1.twinx()
            # sns.set_style("ticks")
            # plt.style.use("science")

            sns.barplot(
                data=subset,
                x=x_level,
                y=metric,
                hue="model_name",
                palette=colors_dict,
            )
            if metric == "logPA":
                plt.gca().invert_yaxis()

            plt.xlabel(x_name)
            plt.ylabel(metric)
            # ax2.set_ylabel(comparison_metric)

            # Legend
            handles, labels1 = plt.gca().get_legend_handles_labels()
            labels = [label_dict[label] + f" ({metric})" for label in labels1]

            plt.legend(handles[::-1], labels[::-1])

            plt.title(f"{level_name} = {value:.4f}")
            fname = osp.join(dirname, f"{metric}_{level}={value:.8f}.png")

            plt.savefig(fname)
            plt.clf()
            plt.close()


def afr_vs_logpa_separate(df: pd.DataFrame, comparison_metric: str = "ASR"):
    """Create and store plots of Linf/Poison ratio vs comparison_metric/logPA.
    Each plot have Linf/Adversarial Ratio on the x axis and two curves on the y
    axis: comparison_metric and LogPA for a (weak/robust) model.
    """
    dirname = osp.join("results", "plots", "dg", "separate")
    os.makedirs(dirname, exist_ok=True)

    pairs = [("adversarial_ratio", "linf"), ("linf", "adversarial_ratio")]
    for levels in tqdm(pairs, total=len(pairs)):
        level, x_level = levels
        level_name, x_name = (
            ("$\ell_\infty$", "Adversarial Ratio")
            if level == "linf"
            else ("Adversarial Ratio", "$\ell_\infty$")
        )
        levels = df[level].unique()
        # import ipdb; ipdb.set_trace()
        for value in tqdm(levels, total=len(levels)):
            # Subset the DataFrame to include only the relevant columns and rows
            level_set = df.loc[
                df[level] == value,
                [
                    "name",
                    "attack_type",
                    "exp_name",
                    "ds2" "logPA",
                    comparison_metric,
                ],
            ]

            dashes_dict = {"Standard": (2, 2), "Engstrom2019Robustness": (2, 2)}
            colors_dict = {
                "Standard": "tab:orange",
                "Engstrom2019Robustness": "tab:blue",
            }
            label_dict = {
                "Standard": "Weak",
                "Engstrom2019Robustness": "Robust",
            }

            # Create a line plot for PGD attack type with Seaborn
            for attack_type in ("BIM", "PGD"):
                for model_name in ("Standard", "Engstrom2019Robustness"):
                    # import ipdb; ipdb.set_trace()
                    subset = level_set[
                        (level_set["attack_type"] == attack_type)
                        & (level_set["model_name"] == model_name)
                    ]

                    _, ax1 = plt.subplots()
                    sns.set_style("whitegrid")
                    ax2 = ax1.twinx()
                    sns.set_style("ticks")

                    ids = subset[x_level].sort_values().index
                    # import ipdb; ipdb.set_trace()
                    ax1.plot(
                        subset[x_level].loc[ids],
                        subset["logPA"].loc[ids],
                        c=colors_dict[model_name],
                        dashes=(None, None),
                        label=f"{model_name} (logPA)",
                        marker="o",
                    )
                    ax2.plot(
                        subset[x_level].loc[ids],
                        subset[comparison_metric].loc[ids],
                        c=colors_dict[model_name],
                        dashes=(2, 2),
                        label=f"{model_name} ({comparison_metric})",
                        marker="X",
                    )
                    # ax1.set_ylim([0, 1])
                    # ax2.set_ylim([0, 1])
                    ax1.set_xlabel(x_name)
                    ax1.set_ylabel("LogPA")
                    ax2.set_ylabel(comparison_metric)
                    handles1, labels1 = ax1.get_legend_handles_labels()
                    handles2, labels2 = ax2.get_legend_handles_labels()
                    handles, labels = handles1 + handles2, labels1 + labels2
                    ax2.legend(handles, labels)
                    ax1.legend().remove()

                    plt.title(
                        f"{attack_type} attack, "
                        f"{label_dict[model_name]} model, "
                        f"{level_name} = {value:.4f}"
                    )

                    fname = osp.join(
                        dirname,
                        f"{attack_type}_{label_dict[model_name]}_{level}={value:.8f}.png",
                    )

                    plt.savefig(fname)
                    plt.clf()
                    plt.close()


"""
if __name__ == "__main__":
    date = "2023-08-15"
    tags = ["diagvib", "500_epochs", "logk_corr", "adam"]
    afr = "pred"

    df = create_dataframe_from_wandb_runs(
        project="adv_pa_new",
        filters={
            "state": "finished",
            "group": "dg_pa_test_euler",
            # "tags": {"$all": ["cifar10", attack]},  # for some reason this does not work
            "$and": [{"tags": tag} for tag in tags],
            "created_at": {"$gte": date},
        },
        date=date,
        afr=afr,
        cache=True,
    )

    # afr_vs_logpa(df, "AFR")
    logpa(df)
    logpa(df)
"""
