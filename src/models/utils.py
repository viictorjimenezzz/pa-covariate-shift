from torch import Tensor


def AFR(y_pred_adv: Tensor, y_pred_clean: Tensor):
    return 1.0 - (y_pred_adv != y_pred_clean).sum() / len(y_pred_adv)
