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
        pa_metric = PosteriorAgreementBase(
                dataset = LogitsDataset([logits_dom1, logits_dom2], y),
                pa_epochs = 50,
                beta0 = 0.0001,
                pairing_strategy = None,
        )

        pa_metric.update(classifier=torch.nn.Identity())
        metric_dict = pa_metric.compute()
    return metric_dict['beta'], metric_dict['logPA'], accuracy

if __name__ == "__main__":
    test_artificial(0.5)