from typing import Tuple

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import pytorch_lightning
import torch
from torch.utils.data import TensorDataset

from pametric.datautils import MultienvDataset, LogitsDataset
from src.data.components.collate_functions import MultiEnv_collate_fn
from pametric.metrics import PosteriorAgreementBase

def test_analytical(prob) -> None:
    """
    Provide metrics (including PA) for a two-class gaussian classifier.
    """
    pytorch_lightning.seed_everything(seed=0, workers=True)
    
    def _generate_logits(y: torch.Tensor) -> torch.Tensor:
        sample_size = y.size(0)
        logits = torch.zeros((sample_size, 2), dtype=torch.float)
        logits[torch.arange(sample_size), y] = 1.0
        return logits

    sample_size = 1000
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

    acc, logPA, beta = torch.zeros(3), torch.zeros(3), torch.zeros(3)
    for indc, classifier_type in enumerate(["random"]):
        preds_dom1 = _generate_preds(y, classifier_type)
        logits_dom1 = _generate_logits(preds_dom1)

        preds_dom2 = _generate_preds(y, classifier_type)
        logits_dom2 = _generate_logits(preds_dom2)

        accuracy = _accuracy(torch.cat([y, y]), torch.cat([preds_dom1, preds_dom2]))
        print(f"ACC '{classifier_type}': {accuracy}".format(accuracy=accuracy))

        pa_metric = PosteriorAgreementBase(
                dataset = LogitsDataset([logits_dom1, logits_dom2], y),
                pa_epochs = 500,
                beta0 = 0.1,
                pairing_strategy = None,
        )

        pa_metric.update(classifier=torch.nn.Identity())
        metric_dict = pa_metric.compute()
        print(f"logPA '{classifier_type}': {metric_dict['logPA']}".format(pa=metric_dict["logPA"]))
        print(f"beta: {metric_dict['beta']}")

        acc[indc] = accuracy
        logPA[indc] = metric_dict['logPA']
        beta[indc] = metric_dict['beta']
    return acc, logPA, beta


if __name__ == "__main__":
    # Fine-grained probability sweep
    prob_tensor = torch.arange(0,1.01,0.01)
    len_prob = prob_tensor.size(0)

    accs = torch.zeros((len_prob, 3))
    logPAs = torch.zeros((len_prob, 3))
    betas = torch.zeros((len_prob, 3))
    for iprob, prob in enumerate(prob_tensor):
        print(f"\n\nPROB = {prob.item()}")
        accs[iprob, :], logPAs[iprob, :], betas[iprob, :] = test_analytical(prob.item())

    import pickle

    results = {
        'accs': accs,
        'logPAs': logPAs,
        'betas': betas,
        'probs': prob_tensor
    }

    file_path = '/cluster/home/vjimenez/adv_pa_new/results/artificial/artificial_results_moreepochs.pkl'
    with open(file_path, 'wb') as f:
        pickle.dump(results, f)