import os
import os.path as osp

import torch
from torch.utils.data import TensorDataset, random_split

from secml.array import CArray
from secml.ml.classifiers import CClassifierPyTorch
from secml.adv.attacks.evasion.foolbox.fb_attacks.fb_ddn_attack import CFoolboxL2DDN
from secml.adv.attacks.evasion.foolbox.fb_attacks.fb_pgd_attack import CFoolboxPGD
from secml.adv.attacks.evasion import CAttackEvasionFoolbox
from foolbox.attacks import LInfFMNAttack
from foolbox.attacks.basic_iterative_method import LinfBasicIterativeAttack
from secml.data.loader import CDataLoaderMNIST
from secml.ml.peval.metrics import CMetricAccuracy
from src.data.components import PairDataset


class AdversarialCIFAR10Dataset(TensorDataset):
    """ Generate adversarially crafted CIFAR10 data, for an image classification 
        problem.
    """
    dset_name: "cifar10"

    def __init__(
            self,
            data_dir: osp.join(".", "data", "datasets"),
            classifier: CClassifierPyTorch,
            poison_ratio: float = 1.,
            cache: bool = False,
            cache_dir: str = osp.join(".", "data", "datasets"),
            attack_type: str = "PGD",
            verbose: bool = False,
            model_name: str = "Standard",
            **kwargs,
    ):

        _, ts = CDataLoaderCIFAR10().load(val_size=0)
        ts.X /= 255.0
        dset1 = TensorDataset(
            torch.from_numpy(ts.X.tondarray()).to(torch.float32),
            torch.from_numpy(ts.Y.tondarray()).to(torch.float32),
        )

        self.classifier = classifier
        config = "_".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        
        fname = osp.join(
            cache_dir,
            f"adv_{dset_name}_model{model_name}_{attack_type}_{config}.pth"
        )

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

            attack = self.get_attack(attack_type)
            adv_Y, adv_scores, adv_ds, adv_f_obj = attack.run(X, Y)

            if verbose:
                print(
                    f"Attack complete! Adversarial {self.dset_name} dataset "
                    "stored in ", fname
                )
            adv_X = self.carray2tensor(adv_ds.X, torch.float32)
            if cache:
                os.makedirs(cache_dir, exist_ok=True)
                torch.save(adv_X.to("cpu"), fname)

        if poison_ratio != 1.:  # TODO: specify samples to be corrupted
            dset_size = X.shape[0]  
            split = int(poison_ratio * dset_size)

            _, unpoison_ids = random_split(
                list(range(dset_size)),
                lengths=(split, dset_size - split)
            )
            
            adv_X[unpoison_ids] = self.carray2tensor(
                X,
                torch.float32
            )[unpoison_ids]

        Y = self.carray2tensor(Y, torch.long)

        super().__init__(adv_X, Y)

    def get_attack(self, attack_type):
        if attack_type == "PGD":
            attack = CFoolboxPGD(
                classifier=classifier,
                abs_stepsize=None,
                **kwargs,
            )
        elif attack_type == "BIM":
            attack = CAttackEvasionFoolbox(
                classifier=classifier,
                fb_attack_class=LinfBasicIterativeAttack,
                **kwargs
            )
        elif attack_type == "FMN":
            attack = CAttackEvasionFoolbox(
                classifier=classifier,
                y_target=None,
                fb_attack_class=LInfFMNAttack,
                **kwargs
            )
        elif attack_type == "L2DDN":
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
        return attack


    def carray2tensor(self, carr, dtype):
        """ Converts a secml CArray to a PyTorch Tensor"""
        return torch.from_numpy(carr.tondarray()).to(dtype)

    def performance_adversarial(self):
        X = CArray(self.tensors[0].to(torch.float64).numpy())
        Y = CArray(self.tensors[1].to(torch.float64).numpy())
        metric = CMetricAccuracy()
        y_pred = self.classifier.predict(X)
        acc = metric.performance_score(y_true=Y, y_pred=y_pred)
        return acc
