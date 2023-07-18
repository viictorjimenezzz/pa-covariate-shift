import os.path as osp

from robustbench import load_model


def get_classifier(
    classifier_name: str,
    checkpoint_dir: str = osp.join("data", "model_checkpoints"),
    **kwargs,
):
    classifier = load_model(
        model_name=classifier_name,
        model_dir=checkpoint_dir,
        **kwargs,
        # dataset="cifar10",
        # threat_model="Linf",
        # model_dir=checkpoint_dir,
    )
    return classifier
