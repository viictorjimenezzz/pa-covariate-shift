from secml.ml.classifiers import CClassifierPyTorch

from secml.adv.attacks.evasion.foolbox.fb_attacks.fb_ddn_attack import (
    CFoolboxL2DDN,
)
from secml.adv.attacks.evasion.foolbox.fb_attacks.fb_pgd_attack import (
    CFoolboxPGD,
)

from secml.adv.attacks.evasion import CAttackEvasionFoolbox
from foolbox.attacks import LInfFMNAttack
from foolbox.attacks.basic_iterative_method import LinfBasicIterativeAttack
from src.data.components.gaussian_attack import GaussianAttack

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
    elif attack_name == "GAUSSIAN":
        attack = GaussianAttack(
            classifier=classifier,
            **kwargs, # the noise_std will be here
        )
    else:
        raise ValueError(
            "Incorrect attack type. Can be one between 'PGD', 'BIM', "
            "'FMN' or 'L2DNN'."
        )
    attack.name = attack_name
    config = "_".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
    attack.info = (
        f"model={classifier.name}" f"_attack={attack_name}" f"_{config}"
        if config
        else ""
    )
    return attack