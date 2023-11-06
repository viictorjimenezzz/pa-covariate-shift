from robustbench import load_model
from src.models.components.jpeg_defense.load_das import load_bpda_model


def get_model(**kwargs):
    """Load a robustbench model and store the name of the model in it."""
    if kwargs["model_name"] == "BPDA":
        model = load_bpda_model(kwargs["jpeg_quality"])
        # monkey patching for integration with CClassifierPyTorch
        # TODO: try to parametrize 10 with number of classes maybe?
        list(model.modules())[-1].out_features = 10
    else:
        model = load_model(
            **kwargs,
            # model_name=model_name,
            # model_dir=model_dir,
            # dataset="cifar10",
            # threat_model="Linf",
            # model_dir=model_dir,
        )

    model.name = kwargs["model_name"]

    return model
