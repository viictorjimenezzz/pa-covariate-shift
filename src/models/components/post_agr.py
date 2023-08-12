from typing import Optional

import torch
import torch.nn.functional as F


class PosteriorAgreementKernel(torch.nn.Module):
    def __init__(self, beta0: Optional[float] = None):
        super().__init__()
        beta0 = torch.tensor(beta0) if beta0 else torch.rand(1)

        if beta0 < 0.0:
            raise ValueError("'beta' must be non-negative.")
        self.beta = torch.nn.Parameter(beta0, requires_grad=True)

        self.register_buffer("log_post", torch.tensor([0.0]))

    def forward(self, preds1, preds2):
        probs1 = F.softmax(self.beta * preds1, dim=1)
        probs2 = F.softmax(self.beta * preds2, dim=1)

        probs_sum = (probs1 * probs2).sum(dim=1)

        # log correction for numerical stability: replace values less than eps
        # with eps, in a gradient compliant way. Replace nans in gradients
        # deriving from 0 * inf
        probs_sum = probs_sum + (probs_sum < 1e-44) * (1e-44 - probs_sum)
        probs_sum.register_hook(torch.nan_to_num)

        self.log_post += torch.log(probs_sum).sum(dim=0)

    def reset(self):
        self.log_post = torch.tensor([0.0], device=self.log_post.device)

    def log_posterior(self):
        return self.log_post

    def posterior(self):
        return torch.exp(self.log_post)
