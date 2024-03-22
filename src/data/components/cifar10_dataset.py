from abc import ABCMeta

import os
import os.path as osp

import torch
from torch.utils.data import TensorDataset, random_split

from secml.array import CArray
from secml.ml.classifiers import CClassifierPyTorch
from secml.adv.attacks.evasion import CAttackEvasion

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
            X = carray2tensor(X, torch.float32)
            if adversarial_ratio == 0.0:
                adv_X = X

            dset_size = X.shape[0]

            split = int(adversarial_ratio * dset_size)
            attack_norms = (adv_X - X).norm(p=float("inf"), dim=1)

            _, unpoison_ids = attack_norms.topk(dset_size - split)

            # remove poison for the largest 1 - adversarial_ratio attacked ones
            adv_X[unpoison_ids] = X[unpoison_ids]

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
