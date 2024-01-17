import torch
from torch import nn
from typing import Optional
import torch.nn.functional as F

class PosteriorAgreementKernel(nn.Module):
    def __init__(self, beta0: Optional[float] = None, device: str = "cpu"):
        super().__init__()
        beta0 = beta0 if beta0 else 1.0
        if beta0 < 0.0:
            raise ValueError("'beta' must be non-negative.")
        
        self.dev = device
        self.beta = torch.nn.Parameter(torch.tensor([beta0], dtype=torch.float), requires_grad=True).to(self.dev)
        self.log_post = torch.tensor([0.0], requires_grad=True).to(self.dev)

    def forward(self, preds1, preds2):
        self.beta.requires_grad_(True)
        self.beta.data.clamp_(min=0.0)
        self.reset()
        
        with torch.set_grad_enabled(True):
            probs1 = F.softmax(self.beta * preds1, dim=1).to(self.dev)
            probs2 = F.softmax(self.beta * preds2, dim=1).to(self.dev)

            probs_sum = (probs1 * probs2).sum(dim=1).to(self.dev)

            # log correction for numerical stability: replace values less than eps
            # with eps, in a gradient compliant way. Replace nans in gradients
            # deriving from 0 * inf
            probs_sum = probs_sum + (probs_sum < 1e-44) * (1e-44 - probs_sum)
            if probs_sum.requires_grad:
                probs_sum.register_hook(torch.nan_to_num)

            #self.log_post += torch.log(probs_sum).sum(dim=0)
            self.log_post = self.log_post + torch.log(probs_sum).sum(dim=0).to(self.dev)
            return -self.log_post

    def evaluate(self, beta_opt, preds1, preds2):
        with torch.set_grad_enabled(False):
            probs1 = F.softmax(beta_opt * preds1, dim=1).to(self.dev)
            probs2 = F.softmax(beta_opt * preds2, dim=1).to(self.dev)
            probs_sum = (probs1 * probs2).sum(dim=1).to(self.dev)
            self.log_post = self.log_post + torch.log(probs_sum).sum(dim=0).to(self.dev)
    
    def reset(self):
        self.log_post = torch.tensor([0.0], requires_grad=True).to(self.dev)

    def log_posterior(self):
        return self.log_post.clone().to(self.dev)

    def posterior(self):
        return torch.exp(self.log_post).to(self.dev)
    
    @property
    def module(self):
        """Returns the kernel itself. It helps the kernel be accessed in both DDP and non-DDP mode."""
        return self