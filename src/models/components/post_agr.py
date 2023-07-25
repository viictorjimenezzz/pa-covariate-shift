from typing import Optional

import torch
import torch.nn.functional as F


class PosteriorAgreementKernel(torch.nn.Module):
    def __init__(self, beta0: Optional[float] = None):
        super().__init__()
        beta0 = torch.tensor(beta0) if beta0 else torch.rand(1.0)

        if beta0 < 0.0:
            raise ValueError("'beta' must be non-negative.")
        self.beta = torch.nn.Parameter(beta0, requires_grad=True)

        self.register_buffer("log_post", torch.tensor([0.0]))

    def forward(self, preds1, preds2):
        probs1 = F.softmax(self.beta * preds1, dim=1)
        probs2 = F.softmax(self.beta * preds2, dim=1)
        self.log_post += torch.log((probs1 * probs2).sum(dim=1)).sum(dim=0)

    def reset(self):
        self.log_post = torch.tensor([0.0], device=self.log_post.device)

    def log_posterior(self):
        return self.log_post

    def posterior(self):
        return torch.exp(self.log_post)
