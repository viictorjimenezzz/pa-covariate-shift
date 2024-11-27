from typing import Union, Optional
import torch
import torch.nn.functional as F
from torch import nn, optim
from pytorch_lightning import LightningModule
from pytorch_lightning.core.optimizer import LightningOptimizer

from src.models.erm import ERM

class AdvModel(nn.Module):
    """
    Incorporates some attributes needed for further processes (callbacks, etc...).
    """
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
        self.net_name = net._get_name().lower() # remove the lower later

    def forward(self, x: torch.Tensor):
        return self.net(x)
    
class AdvLoss:
    def __call__(self, *args, **kwargs):
        return torch.tensor([0.0], requires_grad=True)

class AdvModule(ERM):
    """
    Fake model intended at being used only for a single epoch, without any optimization,
    only to organize callbacks and call the PA metric.
    """

    def __init__(
        self,
        n_classes: int,
        net: nn.Module,

        # For the plots:
        beta_to_plot: Optional[float] = None
    ):
        LightningModule.__init__(self) # I don't want to instantiate ERM
        
        self.beta_to_plot = beta_to_plot

        self.model = AdvModel(net)
        self.loss = AdvLoss()

        self.save_hyperparameters(ignore=["net", "loss"])

        # For the _compute_distributions():
        # self.num_orig_true, self.num_orig_false = 0, 0
        # self.stored_orig_true, self.stored_orig_true_2 = torch.zeros(10).cpu(), torch.zeros(10).cpu()
        # self.stored_orig_gibbs_true, self.stored_orig_gibbs_true_2 = torch.zeros(10).cpu(), torch.zeros(10).cpu()
        # self.stored_orig_false, self.stored_orig_false_2 = torch.zeros(10).cpu(), torch.zeros(10).cpu()
        # self.stored_orig_gibbs_false, self.stored_orig_gibbs_false_2 = torch.zeros(10).cpu(), torch.zeros(10).cpu()

        # self.num_adv_true = 0
        # self.stored_adv_true, self.stored_adv_true_2 = torch.zeros(10).cpu(), torch.zeros(10).cpu()
        # self.stored_adv_gibbs_true, self.stored_adv_gibbs_true_2 = torch.zeros(10).cpu(), torch.zeros(10).cpu()

        # For the _compute_histograms():
        self.stored_orig_true, self.stored_orig_false, self.stored_adv_true = torch.zeros(0).cpu(), torch.zeros(0).cpu(), torch.zeros(0).cpu()
        self.stored_orig_gibbs_true, self.stored_orig_gibbs_false, self.stored_adv_gibbs_true = torch.zeros(0).cpu(), torch.zeros(0).cpu(), torch.zeros(0).cpu()

        # For the _compute_histograms_posteriors():
        # self.stored_x, self.stored_x_gibbs = torch.zeros(0).cpu(), torch.zeros(0).cpu()
        # self.stored_xprime, self.stored_xprime_gibbs = torch.zeros(0).cpu(), torch.zeros(0).cpu()


    @staticmethod
    def softmax(logits):
        return F.softmax(logits, dim=-1)

    @staticmethod
    def gibbs(logits, beta):
        return F.softmax(logits*beta, dim=-1)
    
    def _compute_distributions(self, out: dict):
        beta_opt = self.beta_to_plot

        mask_acc = (out["preds"][:64] == out["targets"][:64]).cpu()
        mask_true = torch.cat([
            mask_acc,
            torch.full((64,), False, dtype=torch.bool),
        ])
        mask_false = torch.cat([
            ~mask_acc,
            torch.full((64,), False, dtype=torch.bool),
        ])
        self.num_orig_true += mask_true.sum().item()
        self.num_orig_false += mask_false.sum().item()

        logits_true = out["logits"][mask_true, :].detach().cpu()
        self.stored_orig_true += self.softmax(logits_true).sort(dim=-1, descending=True)[0].sum(dim=0)
        self.stored_orig_true_2 += self.softmax(logits_true).sort(dim=-1, descending=True)[0].pow(2).sum(dim=0)
        self.stored_orig_gibbs_true += self.gibbs(logits_true, beta_opt).sort(dim=-1, descending=True)[0].sum(dim=0)
        self.stored_orig_gibbs_true_2 += self.gibbs(logits_true, beta_opt).sort(dim=-1, descending=True)[0].pow(2).sum(dim=0)
        
        logits_false = out["logits"][mask_false, :].detach().cpu()
        self.stored_orig_false += self.softmax(logits_false).sort(dim=-1, descending=True)[0].sum(dim=0)
        self.stored_orig_false_2 += self.softmax(logits_false).sort(dim=-1, descending=True)[0].pow(2).sum(dim=0)
        self.stored_orig_gibbs_false += self.gibbs(logits_false, beta_opt).sort(dim=-1, descending=True)[0].sum(dim=0)
        self.stored_orig_gibbs_false_2 += self.gibbs(logits_false, beta_opt).sort(dim=-1, descending=True)[0].pow(2).sum(dim=0)

        del mask_true, mask_false, logits_true, logits_false


        mask_adv = (out["preds"][:64] != out["preds"][64:]).cpu()
        mask_true = torch.cat([
            torch.full((64,), False, dtype=torch.bool),
            mask_acc*mask_adv
        ])
        # mask_false = torch.cat([
        #     torch.full((64,), False, dtype=torch.bool),
        #     ~mask_acc*mask_adv
        # ])
        self.num_adv_true += mask_true.sum().item()

        logits_true = out["logits"][mask_true, :].detach().cpu()
        # logits_false = out["logits"][mask_false, :].detach().cpu()
        self.stored_adv_true += self.softmax(logits_true).sort(dim=-1, descending=True)[0].sum(dim=0)
        self.stored_adv_true_2 += self.softmax(logits_true).sort(dim=-1, descending=True)[0].pow(2).sum(dim=0)
        self.stored_adv_gibbs_true += self.gibbs(logits_true, beta_opt).sort(dim=-1, descending=True)[0].sum(dim=0)
        self.stored_adv_gibbs_true_2 += self.gibbs(logits_true, beta_opt).sort(dim=-1, descending=True)[0].pow(2).sum(dim=0)
        del mask_true, logits_true

        return

    def _compute_histograms(self, out: dict):
        beta_opt = self.beta_to_plot

        mask_acc = (out["preds"][:64] == out["targets"][:64]).cpu()
        mask_true = torch.cat([
            mask_acc,
            torch.full((64,), False, dtype=torch.bool),
        ])
        mask_false = torch.cat([
            ~mask_acc,
            torch.full((64,), False, dtype=torch.bool),
        ])

        logits_true = out["logits"][mask_true, :].detach().cpu()
        print("LOG:", self.softmax(logits_true).sort(dim=-1, descending=True)[0].size())
        self.stored_orig_true = torch.cat([
            self.stored_orig_true,
            self.softmax(logits_true).sort(dim=-1, descending=True)[0]
        ])
        self.stored_orig_gibbs_true = torch.cat([
            self.stored_orig_gibbs_true,
            self.gibbs(logits_true, beta_opt).sort(dim=-1, descending=True)[0]
        ])
        
        
        logits_false = out["logits"][mask_false, :].detach().cpu()
        self.stored_orig_false = torch.cat([
            self.stored_orig_false,
            self.softmax(logits_false).sort(dim=-1, descending=True)[0]
        ])
        self.stored_orig_gibbs_false = torch.cat([
            self.stored_orig_gibbs_false,
            self.gibbs(logits_false, beta_opt).sort(dim=-1, descending=True)[0]
        ])
        del mask_true, mask_false, logits_true, logits_false


        mask_adv = (out["preds"][:64] != out["preds"][64:]).cpu()
        mask_true = torch.cat([
            torch.full((64,), False, dtype=torch.bool),
            mask_acc*mask_adv
        ])

        logits_true = out["logits"][mask_true, :].detach().cpu()
        self.stored_adv_true = torch.cat([
            self.stored_adv_true,
            self.softmax(logits_true).sort(dim=-1, descending=True)[0]
        ])
        self.stored_adv_gibbs_true = torch.cat([
            self.stored_adv_gibbs_true,
            self.gibbs(logits_true, beta_opt).sort(dim=-1, descending=True)[0]
        ])
        del mask_true, logits_true

    def _compute_histograms_posteriors(self, out: dict):
        """
        We will store all values of x' and x''.
        """
        beta_opt = self.beta_to_plot

        logits_x = out["logits"][:64, :].detach().cpu()
        self.stored_x = torch.cat([
            self.stored_x,
            self.softmax(logits_x).sort(dim=-1, descending=True)[0]
        ])
        self.stored_x_gibbs = torch.cat([
            self.stored_x_gibbs,
            self.gibbs(logits_x, beta_opt).sort(dim=-1, descending=True)[0]
        ])

        logits_xprime = out["logits"][64:, :].detach().cpu()
        self.stored_xprime = torch.cat([
            self.stored_xprime,
            self.softmax(logits_xprime).sort(dim=-1, descending=True)[0]
        ])
        self.stored_xprime_gibbs = torch.cat([
            self.stored_xprime_gibbs,
            self.gibbs(logits_xprime, beta_opt).sort(dim=-1, descending=True)[0]
        ])

    def _compute_histograms_posteriors_2(self, out: dict):
        """
        We will only store values for the samples that succeed at misleading the model, before and after the attack.
        """
        beta_opt = self.beta_to_plot

        mask_misleading = (out["preds"][:64] != out["preds"][64:]).cpu()
        mask_x = torch.cat([
            mask_misleading,
            torch.full((64,), False, dtype=torch.bool),
        ])
        y_true = out["targets"][mask_x].detach().cpu()

        logits_x = out["logits"][mask_x, :].detach().cpu()
        self.stored_x = torch.cat([
            self.stored_x,
            self.softmax(logits_x).gather(1, y_true.unsqueeze(1)).squeeze(1)
        ])
        self.stored_x_gibbs = torch.cat([
            self.stored_x_gibbs,
            self.gibbs(logits_x, beta_opt).gather(1, y_true.unsqueeze(1)).squeeze(1)
        ])

        mask_xprime = torch.cat([
            torch.full((64,), False, dtype=torch.bool),
            mask_misleading
        ])
        logits_xprime = out["logits"][mask_xprime, :].detach().cpu()
        self.stored_xprime = torch.cat([
            self.stored_xprime,
            self.softmax(logits_xprime).gather(1, y_true.unsqueeze(1)).squeeze(1)
        ])
        self.stored_xprime_gibbs = torch.cat([
            self.stored_xprime_gibbs,
            self.gibbs(logits_xprime, beta_opt).gather(1, y_true.unsqueeze(1)).squeeze(1)
        ])
    
    def training_step(self, batch: Union[dict, tuple], batch_idx: int):
        out = super().training_step(batch, batch_idx)

        try:
            # self._compute_distributions(out)
            self._compute_histograms(out)
            # self._compute_histograms_posteriors(out)
            # self._compute_histograms_posteriors_2(out)
        except:
            print(f"\nError found at batch {batch_idx}")
        
        return out

    def validation_step(self, batch: Union[dict, tuple], batch_idx: int):
        pass

    def test_step(self, batch: tuple, batch_idx: int):
        pass

    def configure_optimizers(self):
        optimizer = LightningOptimizer(optim.SGD(lr=0.0001, params=self.parameters()))
        return {"optimizer": optimizer}

    def optimizer_step(self, *args, **kwargs):

        for param in self.parameters():
            param.grad = None

        args[-1].__call__()