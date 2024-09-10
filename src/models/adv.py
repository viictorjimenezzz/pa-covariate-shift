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

        self.num_orig_true, self.num_orig_false = 0, 0
        self.stored_orig_true, self.stored_orig_true_2 = torch.zeros(10).cpu(), torch.zeros(10).cpu()
        self.stored_orig_gibbs_true, self.stored_orig_gibbs_true_2 = torch.zeros(10).cpu(), torch.zeros(10).cpu()
        self.stored_orig_false, self.stored_orig_false_2 = torch.zeros(10).cpu(), torch.zeros(10).cpu()
        self.stored_orig_gibbs_false, self.stored_orig_gibbs_false_2 = torch.zeros(10).cpu(), torch.zeros(10).cpu()

        self.num_adv_true = 0
        self.stored_adv_true, self.stored_adv_true_2 = torch.zeros(10).cpu(), torch.zeros(10).cpu()
        self.stored_adv_gibbs_true, self.stored_adv_gibbs_true_2 = torch.zeros(10).cpu(), torch.zeros(10).cpu()

    @staticmethod
    def softmax(logits):
        return F.softmax(logits, dim=-1)

    @staticmethod
    def gibbs(logits, beta):
        return F.softmax(logits*beta, dim=-1)
    
    def _compute_distributions(self, out: dict):
        # FOR THE PLOT -------------------------------------------------------------------------------

        """
        Addepalli2021, 
        PGD, 0.0314
            ar=0.1 >> beta=15.885943
            ar=0.5 >> beta=15.893476
            ar=1.0 >> beta=15.890465.
        PGD, 0.1255
            ar=0.5 >> beta=15.697365
        FMN, ar=1.0 >> beta=6.081631.

        Wang, et al.,
        PGD, 0.0314
            ar=0.1 >> beta=11.257099
            ar=0.5 >> beta=11.259317
            ar=1.0 >> beta=11.244336.
        PGD, 0.1255
            ar=0.5 >> beta=10.858996
        FMN, ar=1.0 >> beta=2.537294.

        Standard,
        PGD, 0.0314
            ar=0.1 >> beta=1.806318
            ar=0.5 >> beta=0.994757
            ar=1.0 >> beta=0.789602.
        PGD,, 0.1255
            ar=0.5 >> beta=0.377333
        FMN, ar=1.0 >> beta=0.655061.

        BPDA,
        PGD, 0.0314
            ar=0.1 >> beta=37.9277
            ar=0.5 >> beta=36.926743
            ar=1.0 >> beta=35.480820.
        PGD, 0.1255
            ar=0.5 >> beta=34.347984
        FMN, ar=1.0 >> beta=19.843315.

        Engstrom,
        PGD, 0.0314
            ar=0.1 >> beta=15.787772
            ar=0.5 >> beta=15.694523
            ar=1.0 >> beta=15.63269.
        PGD, 0.1255
            ar=0.5 >> beta=14.017195
        FMN, ar=1.0 >> beta=2.594392.

        Wong,
        PGD, 0.0314
            ar=0.1 >> beta=15.57553
            ar=0.5 >> beta=15.540545
            ar=1.0 >> beta=15.460098.
        PGD, 0.1255
            ar=0.5 >> beta=13.769010
        FMN, ar=1.0 >> beta=4.596598
        """
        # --------------------------------------------------------------------------------------------
    
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
    
    def training_step(self, batch: Union[dict, tuple], batch_idx: int):
        out = super().training_step(batch, batch_idx)

        try:
            self._compute_distributions(out)
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