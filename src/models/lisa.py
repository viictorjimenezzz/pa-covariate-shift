from src.models.erm import ERM
import torch
from torch import nn, Tensor, optim

from torch.distributions.beta import Beta # to sample mixup weight (lambda)
from torch.distributions.bernoulli import Bernoulli # to select SA strategy (s)

from copy import deepcopy

from src.models.utils import 

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
            loss: nn.Module,
            optimizer: optim.Optimizer,
            scheduler: DictConfig,

            mixup_strategy: str = "mixup",
            ppred: float = 0.5, # probability of LISA-L
            mix_alpha: float = 0.5 # mixup weight
    ):
        super().__init__(n_classes, net, loss, optimizer, scheduler)
        assert mixup_strategy in ["mixup", "cutmix"], "The mixup strategy must be either 'mixup' or 'cutmix'."

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
            Function adapted from the LISA repo to work with pytorch.
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

        if len(x1.shape) == 4:
            l_x = l.unsqueeze(-1).unsqueeze(-1).expand(-1, *x1.shape[1:]).to(self.device)
        else:
            l_x = l.expand(-1, *x1.shape[1:]).to(self.device)
            
        l_y = l.expand(-1, self.hparams.n_classes).to(self.device)

        mixed_x = l_x * x1 + (1 - l_x) * x2
        mixed_y = l_y * y1 + (1 - l_y) * y2
        return mixed_x, mixed_y
    
    def cut_mix(self, mix_alpha: float, x, y):
        def _rand_bbox(size, lam):
            W = size[2]
            H = size[3]
            cut_rat = torch.sqrt(1. - lam).to(self.device)
            cut_w = (W * cut_rat).to(torch.int32).to(self.device)
            cut_h = (H * cut_rat).to(torch.int32).to(self.device)

            # uniform
            cx = torch.randint(0, W, (1,)).item()
            cy = torch.randint(0, H, (1,)).item()

            bbx1 = torch.clamp(cx - torch.div(cut_w, 2, rounding_mode='trunc'), 0, W).to(self.device)
            bby1 = torch.clamp(cy - torch.div(cut_h, 2, rounding_mode='trunc'), 0, H).to(self.device)
            bbx2 = torch.clamp(cx + torch.div(cut_w, 2, rounding_mode='trunc'), 0, W).to(self.device)
            bby2 = torch.clamp(cy + torch.div(cut_h, 2, rounding_mode='trunc'), 0, H).to(self.device)
            return bbx1, bby1, bbx2, bby2
        
        rand_index = torch.randperm(len(y)).to(self.device)
        lam = Beta(mix_alpha, mix_alpha).sample().to(self.device)
        target_a = y
        target_b = y[rand_index]
        bbx1, bby1, bbx2, bby2 = _rand_bbox(x.size(), lam)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))

        return x, lam*target_a + (1-lam)*target_b
    
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


    def selective_augmentation(self, batch: dict):
        """
        Implements mixup and selective augmentation.
        """
        # get data and convert env to tensor
        x = torch.cat([batch[env][0] for env in batch.keys()])
        y = torch.cat([batch[env][1] for env in batch.keys()])
        envs = [
            env
            for env in batch.keys()
            for _ in range(len(batch[env][1]))
        ]

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
                #print("LISA-L, samples with the same label:", len(envs_int[mask]))

                # accumulate indexes wrt all observations
                B1 = torch.cat((B1, torch.index_select(all_inds[mask], 0, B1_lab)))
                B2 = torch.cat((B2, torch.index_select(all_inds[mask], 0, B2_lab)))

        else: # LISA-D
            # group data by environment
            for env in envs_int.unique():
                mask = envs_int.eq(env)
                B1_lab, B2_lab = self.pair_lisa(y[mask]) # indexes wrt mask
                #print("LISA-D, samples with the same domain:", len(y[mask]))

                # accumulate indexes wrt all observations                    
                B1 = torch.cat((B1, torch.index_select(all_inds[mask], 0, B1_lab)))
                B2 = torch.cat((B2, torch.index_select(all_inds[mask], 0, B2_lab)))


        if self.hparams.mixup_strategy == "cutmix":
            joined_indexes = torch.cat([B1, B2]).sort()[0]
            mixed_x, mixed_y = self.cut_mix(
                                        self.hparams.mix_alpha, 
                                        torch.index_select(input=x, dim=0, index=joined_indexes).to(self.device), 
                                        torch.index_select(input=y, dim=0, index=joined_indexes).to(self.device)
            )
            mixed_y = mixed_y.long()
        else: # mixup
            mixed_x, mixed_y = self.mix_up(
                                        self.hparams.mix_alpha, 
                                        torch.index_select(input=x, dim=0, index=B1.sort()[0]).to(self.device), 
                                        self.to_one_hot(torch.index_select(input=y, dim=0, index=B1.sort()[0]), self.hparams.n_classes), 
                                        torch.index_select(input=x, dim=0, index=B2.sort()[0]).to(self.device),
                                        self.to_one_hot(torch.index_select(input=y, dim=0, index=B2.sort()[0]), self.hparams.n_classes)
            )
            mixed_y = self.from_one_hot(mixed_y)

        return mixed_x, mixed_y

    def _extract_batch(self, batch: dict):
        return self.selective_augmentation(batch)