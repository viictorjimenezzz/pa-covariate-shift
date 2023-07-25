from abc import ABCMeta

import os
import os.path as osp

import torch
from torch.utils.data import TensorDataset, random_split

from secml.array import CArray
from secml.ml.classifiers import CClassifierPyTorch
from secml.adv.attacks.evasion import CAttackEvasion
from secml.adv.attacks.evasion.foolbox.fb_attacks.fb_ddn_attack import (
    CFoolboxL2DDN,
)
from secml.adv.attacks.evasion.foolbox.fb_attacks.fb_pgd_attack import (
    CFoolboxPGD,
)

from secml.adv.attacks.evasion import CAttackEvasionFoolbox
from foolbox.attacks import LInfFMNAttack
from foolbox.attacks.basic_iterative_method import LinfBasicIterativeAttack
from secml.data.loader import CDataLoaderCIFAR10
from secml.ml.peval.metrics import CMetricAccuracy

from src.data.utils import carray2tensor


class AdversarialCIFAR10Dataset(TensorDataset):
    """Generate adversarially crafted CIFAR10 data, for an image
    classification problem.
    """

    dset_name: str = "cifar10"
    dset: ABCMeta = CDataLoaderCIFAR10
    dset_shape: tuple = (3, 32, 32)

    def __init__(
        self,
        attack: CAttackEvasion,
        classifier: CClassifierPyTorch,
        data_dir: str = osp.join(".", "data", "datasets"),
        checkpoint_fname: str = "checkpoint.pt",
        adversarial_ratio: float = 1.0,
        verbose: bool = False,
        cache: bool = False,
    ):
        _, ts = self.dset().load(val_size=0)
        X, Y = ts.X / 255.0, ts.Y

        self.attacked_classifier = classifier

        fname = osp.join(data_dir, checkpoint_fname)
        if cache and osp.exists(fname):
            if verbose:
                print(
                    f"Loaded found Adversarial {self.dset_name} dataset "
                    f"in {fname}"
                )
            adv_X = torch.load(fname)
        else:
            if verbose:
                print("Attack started...")

            adv_Y_pred, adv_scores, adv_ds, adv_f_obj = attack.run(X, Y)

            if verbose:
                print(
                    f"Attack complete! Adversarial {self.dset_name} dataset "
                    "stored in ",
                    fname,
                )
            adv_X = carray2tensor(adv_ds.X, torch.float32)
            if cache:
                os.makedirs(data_dir, exist_ok=True)
                torch.save(adv_X.to("cpu"), fname)

        if adversarial_ratio != 1.0:  # TODO: specify samples to be corrupted
            dset_size = X.shape[0]
            split = int(adversarial_ratio * dset_size)

            _, unpoison_ids = random_split(
                list(range(dset_size)), lengths=(split, dset_size - split)
            )

            adv_X[unpoison_ids] = carray2tensor(X, torch.float32)[unpoison_ids]

        adv_X = adv_X.reshape(-1, *self.dset_shape)
        Y = carray2tensor(Y, torch.long)

        super().__init__(adv_X, Y)

    def performance_adversarial(self):
        X = CArray(self.tensors[0].to(torch.float64).numpy())
        Y = CArray(self.tensors[1].to(torch.float64).numpy())
        metric = CMetricAccuracy()
        y_pred = self.classifier.predict(X)
        acc = metric.performance_score(y_true=Y, y_pred=y_pred)
        return acc


def get_attack(attack_name: str, classifier: CClassifierPyTorch, **kwargs):
    """Retrieve the attack and store its name."""
    if attack_name == "PGD":
        attack = CFoolboxPGD(
            classifier=classifier,
            abs_stepsize=None,
            **kwargs,
        )
    elif attack_name == "BIM":
        attack = CAttackEvasionFoolbox(
            classifier=classifier,
            fb_attack_class=LinfBasicIterativeAttack,
            **kwargs,
        )
    elif attack_name == "FMN":
        attack = CAttackEvasionFoolbox(
            classifier=classifier,
            y_target=None,
            fb_attack_class=LInfFMNAttack,
            **kwargs,
        )
    elif attack_name == "L2DDN":
        attack = CFoolboxL2DDN(
            classifier=classifier,
            abs_stepsize=None,
            **kwargs,
        )
    else:
        raise ValueError(
            "Incorrect attack type. Can be one between 'PGD', 'BIM', "
            "'FMN' or 'L2DNN'."
        )
    attack.name = attack_name
    config = "_".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
    attack.info = (
        f"model={classifier.name}_"
        f"attack={attack_name}_"
        f"{config}"
    )
    return attack
