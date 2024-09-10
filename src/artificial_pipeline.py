from typing import Tuple

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import pytorch_lightning
import torch
from torch.utils.data import TensorDataset

from pametric.datautils import MultienvDataset, LogitsDataset
from src.data.components.collate_functions import MultiEnv_collate_fn
from pametric.metrics import PosteriorAgreementBase

def test_artificial(prob: float) -> None:
    """
    Provide metrics (including PA) of a constant, random and perfect classifier.
    """
    pytorch_lightning.seed_everything(seed=0, workers=True)
    
    def _generate_logits(y: torch.Tensor) -> torch.Tensor:
        # Difference in the logits:
        logit_diff = 1.0

        sample_size = y.size(0)
        logits = -(logit_diff/2.0)*torch.ones((sample_size, 2), dtype=torch.float)
        logits[torch.arange(sample_size), y] = logit_diff/2.0
        return logits

    sample_size = 100
    y = torch.bernoulli(torch.ones(sample_size), prob).to(dtype=torch.long)

    def _generate_preds(y: torch.Tensor, classifier_type: str) -> torch.Tensor:
        assert classifier_type in ["perfect", "random", "constant"]

        if classifier_type == "perfect":
            return y
        elif classifier_type == "random":
            return y[torch.randperm(y.size(0))]
        else: # constant
            return torch.zeros_like(y)
        
    def _accuracy(y: torch.Tensor, preds: torch.tensor) -> float:
        return (y == preds).float().mean().item()

    for classifier_type in ["random"]: #, "random", "constant"]:
        preds_dom1 = _generate_preds(y, classifier_type)
        logits_dom1 = _generate_logits(preds_dom1)

        preds_dom2 = _generate_preds(y, classifier_type)
        logits_dom2 = _generate_logits(preds_dom2)

        accuracy = _accuracy(torch.cat([y, y]), torch.cat([preds_dom1, preds_dom2]))
        # print(f"\nACC '{classifier_type}': {accuracy:.2f}".format(accuracy=accuracy))

        pa_metric = PosteriorAgreementBase(
                dataset = LogitsDataset([logits_dom1, logits_dom2], y),
                pa_epochs = 50,
                beta0 = 0.0001,
                pairing_strategy = None,
        )

        pa_metric.update(classifier=torch.nn.Identity())
        metric_dict = pa_metric.compute()
        # print(f"logPA '{classifier_type}': {metric_dict['logPA']:.4f}".format(pa=metric_dict["logPA"]))
        # print(f"beta: {metric_dict['beta']}")
    return metric_dict['beta'], metric_dict['logPA'], accuracy


# if __name__ == "__main__":
    """
    Show as you increase N, that changing the probability yields the same PA for constant and
    perfect (because they are robust), while not for the random one.

    If accuracy = 0.7 (for instance), then you can divide the samples in two groups. One group with
    paired aligned-misaligned containing 0.6*N of the samples (the 0.3*N wrong with some 0.3*N right), and
    a group with the 0.4 right.

    import torch
    import torch.nn.functional as F

    N = ??
    beta = ??

    probdist = F.softmax(beta * torch.tensor([[1.0, 0.0]]), dim=1)
    delta = abs(probdist[0][0].item() - probdist[0][1].item())

    EXAMPLE: N = 1000
    p=0.4 >> acc = 0.53, beta=0.36, logPA = -690.82 >> ~ -0.94*N*delta**4 + 0.06*N*delta**2 = 0.95
    p=0.8 >> acc = 0.69, beta=1.41, logPA = -619.17 >> ~ -0.62*N*delta**4 + 0.38*N*delta**2 = 55.8

    Asi que la diferencia entre los dos deberia estar en ~ 55, y hemos obtenido ~ 70.

    EXAMPLE: N = 100
    p=0.2 >> acc = 0.62, beta=0.71, logPA = -68.59 >> ~ -0.76*N*delta**4 + 0.24*N*delta**2 = 1.76
    p=0.8 >> acc = 0.72, beta=1.59, logPA = -59.3 >> ~ -0.56*N*delta**4 + 0.44*N*delta**2 = 8.53

    Asi que la diferencia deberia estar cerca de ~7 y tenemos ~9.
    """

    # beta, logPA, accuracy = test_artificial(0.5)


    # for prob in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    # for prob in [0.5]:
    #     print(f"\nProbability: {prob}")
    #     test_artificial(prob)