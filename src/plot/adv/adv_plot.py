import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.plot.adv.utils import create_dataframe_from_wandb_runs

attack = "GAUSSIAN" # "PGD", "FMN
date = "2024-01-15"
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
    project="malvai/cov_pa",
    attack=attack,
    date=date,
    filters={
        "state": "finished",
        "group": "adv_pa_cifar",
        # "tags": {"$all": ["cifar10", attack]},  # for some reason this does not work
        "$and": [{"tags": tag} for tag in tags],
        "created_at": {"$gte": date},
    },
    afr=afr,
    cache=True,
)
df.loc[df["adversarial_ratio"].eq(0.0), "logPA"] = 0.0

from src.plot.adv.pgd import table, curves
#from src.plot.adv.fmn import table, curves

table(df)
curves(df, "logPA", attack_name = attack)
#curves(df, "logPA")