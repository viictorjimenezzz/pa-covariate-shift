from src.models.erm_module import ERM
import torch
from torch import nn, Tensor, optim
from typing import List, Callable

from torch.distributions.beta import Beta # to sample mixup weight (lambda)
from torch.distributions.bernoulli import Bernoulli # to select SA strategy (s)

class LISA(ERM):
    """
    Implements selective augmentation on top of ERM.
    
    Instead of performing mixup in the collate function for both intra-label or intra-domain SA, 
    the mixup is performed on the fly in the module for the selected SA strategy s.

    It could be a problem if B_env1 and B_env2 do not share a single observation with the same label.
    So far, batch_size=64 does not seem to be a problem if samples are permuted randomly before going into batches.

    Args:
        n_classes: Number of classes.
        net: Network to be trained.
        optimizer: Desired torch.optim.Optimizer.
        scheduler: Desired torch.optim.lr_scheduler.
        ppred: Probability of LISA-L.
        mix_alpha: Mixup weight.
    """
    def __init__(
            self,
            n_classes: int,
            net: nn.Module,
            optimizer: optim.Optimizer,
            scheduler: optim.lr_scheduler,
            ppred: float = 0.5, # probability of LISA-L
            mix_alpha: float = 0.5 # mixup weight
    ):
        super().__init__(n_classes, net, optimizer, scheduler)
        self.save_hyperparameters(ignore=['net'])

    def to_one_hot(self, target, C):
        """Converts a tensor of labels into a one-hot-encoded tensor."""

        one_hot = torch.zeros(target.size(0), C).to(self.device)
        one_hot.scatter_(1, target.unsqueeze(1), 1)
        return one_hot.to(self.device)

    def from_one_hot(self, one_hot):
        """Converts one-hot-encoded tensor to a tensor of labels."""
        
        _, indices = torch.max(one_hot, 1)
        return indices.to(self.device)

    def mix_up(self, mix_alpha: float, x: Tensor, y: Tensor, x2: Tensor = None, y2: Tensor = None):
        """y_1 and y_2 should be one-hot encoded.
            Function adapted from the LISA repo to work only with pytorch.
        """

        if x2 is None:
            idxes = torch.randperm(len(x)).to(self.device)
            x1 = x
            x2 = x[idxes]
            y1 = y
            y2 = y[idxes]
        else:
            x1 = x
            y1 = y

        bsz = len(x1)
        a = torch.full((bsz, 1), mix_alpha).to(self.device)
        l = Beta(a, a).sample().to(self.device)

        if (len(a) == 0) or (len(l) == 0) or (len(x1) == 0):
            raise NotImplementedError(str([str(i) for i in [len(a), len(l), len(x1)]]))

        if len(x1.shape) == 4:
            l_x = l.unsqueeze(-1).unsqueeze(-1).expand(-1, *x1.shape[1:]).to(self.device)
        else:
            l_x = l.expand(-1, *x1.shape[1:]).to(self.device)
            
        l_y = l.expand(-1, self.hparams.n_classes).to(self.device)

        mixed_x = l_x * x1 + (1 - l_x) * x2
        mixed_y = l_y * y1 + (1 - l_y) * y2
        return mixed_x, mixed_y
    
    def pair_lisa(self, cat: Tensor):
        """
        It pairs observations with different attributes. It is important to note that the
        number of non-paired observations is not maximized, as in such case we would require that
        the categories with the most number of observations are paired first, and that could impose a
        misrepresentation of data of less represented categories.
        """

        cat_original = cat.clone().detach()
        ind_original = torch.arange(len(cat_original)).to(self.device)
        B_1 = torch.empty(0, dtype=torch.int8).to(self.device)
        B_2 = torch.empty(0, dtype=torch.int8).to(self.device)
        for cat_it in range(len(torch.unique(cat_original))-1):
            # new category list
            remove_indexes = torch.cat((B_1, B_2)).to(dtype=torch.long)
            mask = torch.ones(cat_original.size(0), dtype=torch.bool).to(self.device)
            mask[remove_indexes] = False
            cat = cat_original[mask]

            # select smallest category
            unique_elements, counts = torch.unique(cat, return_counts=True)
            ordered_cats = unique_elements[torch.argsort(-counts)]
            smallest_cat = ordered_cats[-1].item()

            # select indexes to pair
            ind_smallcat = ind_original[mask & cat_original.eq(smallest_cat)].to(self.device)
            ind_restcat = ind_original[mask & cat_original.ne(smallest_cat)].to(self.device)
            ind_choice = ind_restcat[torch.randperm(len(ind_restcat)).to(self.device)[:len(ind_smallcat)]].to(self.device)

            # pair with the rest
            B_1 = torch.cat((B_1, ind_smallcat))
            B_2 = torch.cat((B_2, ind_choice))

        return  B_1, B_2 # indexes


    def model_step(self, batch):
        """
        Implements mixup and selective augmentation.
        """
        # get data and convert env to tensor
        x, y, envs = batch
        all_inds = torch.arange(len(envs)).to(self.device)
        env_to_int = {item: i for i, item in enumerate(set(envs))}
        envs_int = torch.tensor([env_to_int[item] for item in envs], dtype=torch.int8).to(self.device) # envs in a tensor

        # select LISA strategy      
        if len(set(envs)) == 1: # only one environment, so only LISA-D
            s = torch.tensor([0.0]).to(self.device)
        else:
            s = Bernoulli(torch.tensor([self.hparams.ppred]).to(self.device)).sample() # select strategy

        # build datasets to be mixed
        B1 = torch.empty(0, dtype=torch.long).to(self.device)
        B2 = torch.empty(0, dtype=torch.long).to(self.device)
        if int(s.item()) == 1: # LISA-L
            # group data by label
            for label in y.unique():
                mask = y.eq(label)
                B1_lab, B2_lab = self.pair_lisa(envs_int[mask]) # indexes wrt mask

                # accumulate indexes wrt all observations
                B1 = torch.cat((B1, torch.index_select(all_inds[mask], 0, B1_lab)))
                B2 = torch.cat((B2, torch.index_select(all_inds[mask], 0, B2_lab)))

        else: # LISA-D
            # group data by environment
            for env in envs_int.unique():
                mask = envs_int.eq(env)
                B1_lab, B2_lab = self.pair_lisa(y[mask]) # indexes wrt mask

                # accumulate indexes wrt all observations
                B1 = torch.cat((B1, torch.index_select(all_inds[mask], 0, B1_lab)))
                B2 = torch.cat((B2, torch.index_select(all_inds[mask], 0, B2_lab)))

        # mixup
        mixed_x, mixed_y = self.mix_up(self.hparams.mix_alpha, 
                                            torch.index_select(input=x, dim=0, index=B1).to(self.device), 
                                            self.to_one_hot(torch.index_select(input=y, dim=0, index=B1), self.hparams.n_classes), 
                                            torch.index_select(input=x, dim=0, index=B2).to(self.device),
                                            self.to_one_hot(torch.index_select(input=y, dim=0, index=B2), self.hparams.n_classes))

        return mixed_x, self.from_one_hot(mixed_y)


    def training_step(self, batch, batch_idx):
        mixed_x, mixed_y = self.model_step(batch)
        logits = self.model(mixed_x)

        return {"logits": logits, "targets": mixed_y}
    
    def validation_step(self, batch, batch_idx):
        mixed_x, mixed_y = self.model_step(batch)
        logits = self.model(mixed_x)

        return {"logits": logits, "targets": mixed_y}
    

class LISAMnist(LISA):
    def __init__(
        self,
        n_classes: int,
        net: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler,
        ppred: float = 0.5, # probability of LISA-L
        mix_alpha: float = 0.5
    ):
        super().__init__(n_classes, net, optimizer, scheduler, ppred, mix_alpha)

        self.model = net
        self.loss = nn.CrossEntropyLoss()
