LIST_PASPECIFIC_METRICS = ["logPA", "beta", "acc_pa", "AFR_true", "AFR_pred", "CD", "MMD", "FID", "CS", "KL", "W", "oracle"]
COLORS_DICT = {
    "ERM": "tab:blue",
    "IRM": "tab:orange",
    "LISA": "tab:green"
}
LABELS_DICT = {
    "ERM": "Vanilla ERM",
    "IRM": "Arjovsky et al.",
    "LISA": "Yao et al."
}

METRIC_DICT = {
    "beta": r"$\beta$",
    "FID": "FID",
    "CD": "Centroid Distance",
    "MMD": "Maximum Mean Discrepancy",
    "CS": "Cosine Similarity",
    "KL": "Kullback-Leibler Distance",
    "W": "Wasserstein Distance",
    "acc_pa": "Joint Accuracy",
    "AFR_true": "AFR (T)",
    "AFR_pred": "AFR (P)",
    "val/acc": "Accuracy (V)"
}

DG_MODELSELECTION = [
    "hue_idval_17",
    "hue_idval_49",
    "hue_idval",
    "hue_oodval",
    "hue_mixval",
    "hue_maxmixval",
    "pos_idval_17",
    "pos_idval_49",
    "pos_idval",
    "pos_oodval",
    "pos_mixval",
    "pos_maxmixval",
]


DG_DATASHIFT = [
    "paper",
    "ZGO_hue_3",
    "ZSO_hue_3",
    "CGO_1_hue",
    "CGO_2_hue"
    "CGO_3_hue"
]