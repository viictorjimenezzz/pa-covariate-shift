import os
import os.path as osp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import numpy as np

from src.plot.adv import DASHES_DICT, COLORS_DICT, LABEL_DICT, YEARS
from src.plot.adv.utils import create_dataframe_from_wandb_runs


def curves(df: pd.DataFrame, metric: str = "logPA") -> None:
    """Create and store plots of Linf/Poison ratio vs comparison_metric/logPA.
    Each plot have Linf/Adversarial Ratio on the x axis and four curves on the y
    axis: comparison_metric for a weak (Standard) and robust model,logPA for a
    weak and robust model.

    df (pandas.DataFrame): the DataFrame generated by create_dataframe
    comparison_metric (str): the metric used for the comparison with logPA
    """

    for i in range(2): # First one gets small font for some reason
        dirname = osp.join("results", "plots", "adv", "PA" if metric == "logPA" else "AFR")
        os.makedirs(dirname, exist_ok=True)

        x_var = "adversarial_ratio"
        x_label = "Adversarial Ratio"

        fontname = "DejaVu Serif"
        _ = fm.findfont(fm.FontProperties(family=fontname))
        # Subset the DataFrame to include only the relevant columns and rows
        level_set = df.loc[
            :,  # ~df["adversarial_ratio"].eq(0.0),
            [
                "attack_name",
                "model_name",
                "adversarial_ratio",
                "logPA",
                "AFR",
            ],
        ]

        # Create a line plot for PGD attack type with Seaborn
        attack_name = "FMN"
        subset = level_set[level_set["attack_name"] == attack_name]
        _, ax = plt.subplots(figsize=(2 * 3.861, 2 * 2.7291))
        sns.set(font_scale=1.9)
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = fontname
        sns.set_style("ticks")

        # Divide by the cardinality of cifar10
        #subset["logPA"] = subset["logPA"]/10000.0

        sns.lineplot(
            data=subset,
            ax=ax,
            x=x_var,
            y=metric,
            hue="model_name",
            style="model_name",
            palette=COLORS_DICT,
            dashes=False,
            marker="o",
            linewidth=3,
        )

        ax.minorticks_on()
        ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        plt.xticks(rotation=45)
        ax.tick_params(axis="both", which="both", direction="in")
        xticks_font = fm.FontProperties(family=fontname)
        for tick in ax.get_xticklabels():
            tick.set_fontproperties(xticks_font)

        ax.grid(linestyle="--")

        ax.set_xlabel(x_label, fontname=fontname)
        #ax.set_ylabel(r"$10^{4} \cdot $ PA" if metric == "logPA" else "AFR", fontname=fontname)
        ax.set_ylabel("PA" if metric == "logPA" else "AFR", fontname=fontname)

        # Legend
        handles, labels = ax.get_legend_handles_labels()
        labels = [LABEL_DICT[label] for label in labels]

        # sort labels and handles
        ids = sorted(range(len(labels)), key=YEARS.__getitem__)

        #ids = [5, 0, 2, 4, 3, 1]
        labels = [labels[i] for i in ids]
        handles = [handles[i] for i in ids]

        ax.legend(
            handles,
            labels,
            handlelength=0.5,
            prop={"family": fontname, "size": 16},
        )
        
        # Set the y-ticks you want
        # ax.set_yticks([0, -1000, -2000, -4000, -6000, -8000, -10000])  # Adjust these values as needed
        # def format_tick(val, pos=None):
        #     if val == 0:
        #         return r'$0$'
        #     exp = int(np.log10(abs(val)))
        #     if val == -2000:
        #         return r'$-2 \cdot 10^{{{}}}$'.format(exp)
        #     elif val == -4000:
        #         return r'$-4 \cdot 10^{{{}}}$'.format(exp)
        #     elif val == -6000:
        #         return r'$-6 \cdot 10^{{{}}}$'.format(exp)
        #     elif val == -8000:
        #         return r'$-8 \cdot 10^{{{}}}$'.format(exp)
        #     else:
        #         return r'$-10^{{{}}}$'.format(exp)
        # ax.yaxis.set_major_formatter(plt.FuncFormatter(format_tick))
        # ax.set_ylim(-8000, None)

        ax.set_title(f"{attack_name} attack", fontname=fontname)

        plt.tight_layout()
        fname = osp.join(dirname, f"{attack_name}.pdf")
        plt.savefig(fname)
        plt.clf()
        plt.close()


def afr_vs_logpa(df: pd.DataFrame, comparison_metric: str = "AFR"):
    """Create and store plots of Linf/Poison ratio vs comparison_metric/logPA.
    Each plot have Linf/Adversarial Ratio on the x axis and four curves on the y
    axis: comparison_metric for a weak (Standard) and robust model,logPA for a
    weak and robust model.

    df (pandas.DataFrame): the DataFrame generated by create_dataframe
    comparison_metric (str): the metric used for the comparison with logPA
    """
    dirname = osp.join("results", "plots", "adv", "joint")
    os.makedirs(dirname, exist_ok=True)

    x_var = "adversarial_ratio"
    x_label = "Adversarial Ratio"

    fontname = "Times New Roman"
    _ = fm.findfont(fm.FontProperties(family=fontname))
    # Subset the DataFrame to include only the relevant columns and rows
    level_set = df.loc[
        :,
        [
            "attack_name",
            "model_name",
            "adversarial_ratio",
            "logPA",
            comparison_metric,
        ],
    ]

    # Create a line plot for PGD attack type with Seaborn
    attack_name = "FMN"
    subset = level_set[level_set["attack_name"] == attack_name]
    _, ax1 = plt.subplots(figsize=(2 * 3.861, 2 * 2.7291))
    sns.set(font_scale=1.9)
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = fontname
    sns.set_style("ticks")

    ax2 = ax1.twinx()

    sns.lineplot(
        data=subset,
        ax=ax1,
        x=x_var,
        y="logPA",
        hue="model_name",
        style="model_name",
        palette=COLORS_DICT,
        dashes=False,
        marker="o",
        # linewidth=2,
    )

    sns.lineplot(
        data=subset,
        ax=ax2,
        x=x_var,
        y="AFR",
        hue="model_name",
        style="model_name",
        palette=COLORS_DICT,
        dashes=DASHES_DICT,
        marker="X",
        # linewidth=2,
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

    ax1.set_xlabel(x_label, fontname=fontname)
    ax1.set_ylabel("LogPA", fontname=fontname)
    ax2.set_ylabel(comparison_metric, fontname=fontname)

    # Legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    labels1 = [LABEL_DICT[label] + " (logPA)" for label in labels1]
    handles2, labels2 = ax2.get_legend_handles_labels()
    labels2 = [LABEL_DICT[label] + " (AFR)" for label in labels2]
    labels = labels1 + labels2
    handles = handles1 + handles2

    # sort labels and handles
    ids = sorted(range(len(labels)), key=YEARS.__getitem__)
    labels = [labels[i] for i in ids]
    handles = [handles[i] for i in ids]

    ax2.legend(handles, labels)
    # sns.move_legend(ax2, "upper right")
    ax1.legend().remove()

    ax1.set_title(f"{attack_name} attack", fontname=fontname)

    plt.tight_layout()
    fname = osp.join(dirname, f"{attack_name}.pdf")
    plt.savefig(fname)
    plt.clf()
    plt.close()


def table(df: pd.DataFrame) -> None:
    dset = df.loc[
        df["adversarial_ratio"] == 1.0,
        [
            "model_name",
            "logPA",
            "AFR",
        ],
    ]
    dset["logPA"] = dset["logPA"].apply(lambda x: int(round(x, 0)))
    dset = dset.replace(LABEL_DICT)
    dset = pd.melt(dset, id_vars=["model_name"], value_vars=["logPA", "AFR"])
    # dset = dset.sort_values(by="linf")
    dset = dset.pivot(index="model_name", columns=["variable"], values="value")
    dset.index.name = "Models"
    print(
        dset.to_latex(
            float_format="{:.0f}".format,
            escape=False,
            sparsify=True,
            multirow=True,
            multicolumn=True,
            multicolumn_format="c",
            caption="\small PA vs. AFR, model discriminability.",
            label="tab:logpa_pgd",
            position="t",
        )
    )


if __name__ == "__main__":
    attack = "FMN"
    date = "2023-08-02"
    tags = [
        "cifar10",
        attack,
        "adam",
        "1000_steps",
        "500_epochs",
        "order_by_attack",
    ]
    afr = "pred"

    df = create_dataframe_from_wandb_runs(
        project="adv_pa_new",
        attack=attack,
        date=date,
        filters={
            "state": "finished",
            "group": "adversarial",
            # "tags": {"$all": ["cifar10", attack]},  # for some reason this does not work
            "$and": [{"tags": tag} for tag in tags],
            "created_at": {"$gte": date},
        },
        afr=afr,
        cache=True,
    )

    df.loc[df["adversarial_ratio"].eq(0.0), "logPA"] = 0.0
    table(df)
    # curves(df, "logPA")
    # curves(df, "logPA")
