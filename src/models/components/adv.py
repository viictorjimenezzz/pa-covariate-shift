from robustbench import load_model


def get_model(**kwargs):
    """Load a robustbench model and store the name of the model in it."""
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
