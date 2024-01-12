import torch
from typing import Union
import warnings
from torch.utils.data.distributed import DistributedSampler
from src.data.components import MultienvDataset, LogitsDataset

class PosteriorAgreementSampler(DistributedSampler):
    def __init__(self, dataset: Union[MultienvDataset, LogitsDataset], *args, **kwargs):
        """
        - If the dataset contains only one environment, the metric will expect two classifiers in the .update() method.
        - If the dataset contains more than one environment, those not used (at least one if two classifiers are provided) will be used for validation.
        """

        if not (isinstance(dataset, MultienvDataset) or isinstance(dataset, LogitsDataset)):
            warnings.warn("The dataset must be a MultienvDataset to work with the PA metric.")
        
        self.num_envs = dataset.num_envs
        original_dset_list = dataset.dset_list
        if dataset.num_envs == 1:
            warnings.warn("Only one environment was found in the dataset. The PA metric will expect two classifiers in the .update() method.")

        # Build two first environments to be PA pairs.
        if self.num_envs >= 2:
            dataset.num_envs = 2
            dataset.dset_list = dataset.dset_list[:2]
            dataset.permutation = self._pair_optimize(dataset)

            # Add additional environments adjusted to the first two.
            if self.num_envs > 2:
                new_permutations = [None]*(self.num_envs-2)
                new_nsamples = dataset.__len__() # samples after pairing (new permutation has been applied)
                new_labels = dataset.__getlabels__(list(range(new_nsamples)))[0] # labels of first environment (idem second)
                add_dataset = MultienvDataset(original_dset_list[2:])
                print("len ", add_dataset.__len__())
                add_labels = add_dataset.__getlabels__(list(range(add_dataset.__len__()))) # labels of the rest of environments
                for i in range(self.num_envs-2):
                    new_permutations[i] = self._pair_validate(new_labels, add_labels[i])

                filtered = torch.tensor([new_permutations[i] != None for i in range(self.num_envs-2)])
                dataset.num_envs = 2 + filtered.sum().item()
                dataset.dset_list = original_dset_list[:2] + [original_dset_list[2+i] for i in range(len(filtered)) if filtered[i].item()]
                dataset.permutation = dataset.permutation + [newperm for newperm, flag in zip(new_permutations, filtered) if flag]

                print(dataset)
                print(dir(dataset))


        super().__init__(dataset, *args, **kwargs)

    def _pair_optimize(self, dataset: MultienvDataset):
        """
        Generates permutations for the first pair of environments so that their labels are correspondent.
        """
        n_samples = dataset.__len__()
        inds = torch.arange(n_samples)
        labels_list = dataset.__getlabels__(inds.tolist())[:2] # only the first two environments

        # IMPORTANT: If the data is already paired, it could mean that not only the labels are paired but also the samples.
        # In such case, we don't want to touch it.
        if torch.equal(labels_list[0], labels_list[1]):
            return [inds, inds]

        unique_labs = [labels.unique() for labels in labels_list] 
        common_labs = unique_labs[0][torch.isin(unique_labs[0], unique_labs[1])] # labels that are common to both environments

        final_inds = [[], []]
        for lab in list(common_labs):
            inds_mask = [inds[labels_list[i].eq(lab)] for i in range(2)] # indexes for every label
            if len(inds_mask[0]) >= len(inds_mask[1]):
                final_inds[0].append(inds_mask[0][:len(inds_mask[1])])
                final_inds[1].append(inds_mask[1])
            else:
                final_inds[0].append(inds_mask[0])
                final_inds[1].append(inds_mask[1][:len(inds_mask[0])])

        return [torch.cat(final_inds[i]).tolist() for i in range(2)]
    
    def _pair_validate(self, labels: torch.Tensor, labels_add: torch.Tensor):
        """
        Generates permutations for additional validation environments so that their labels are correspondent to the PA pair.
        If the number of observations for certain labels is not enough, the samples are repeated.
        If there are not observations associated with specific reference labels, the environment will be discarded.
        """
        if torch.equal(labels, labels_add):
            return torch.arange(len(labels)).tolist() # do not rearrange if labels are already equal
        
        unique, counts = labels.unique(return_counts=True)
        sorted_values, sorted_indices = torch.sort(labels)
        unique_add, counts_add = labels_add.unique(return_counts=True)
        sorted_values_add, sorted_indices_add = torch.sort(labels_add)

        permuted = []
        for i in range(len(unique)):
            pos_add = (unique_add==unique[i].item()).nonzero(as_tuple=True)[0]
            if len(pos_add) == 0: # it means that that the label is not present in the second tensor
                warnings.warn("The label " + str(unique[i].item()) + " is not present in the tensor. Pairig is impossible, so the environment will not be used.")
                return None
            else:
                num = counts[i] # elements in the reference
                num_add = counts_add[pos_add.item()] # elements in the second tensor 
                diff = num_add - num
                vals_add = sorted_indices_add[counts_add[:pos_add].sum(): counts_add[:pos_add+1].sum()] # indexes of the second tensor
                if diff >= 0: # if there are enough in the second tensor, we sample without replacement
                    permuted.append(vals_add[torch.randperm(num_add)[:num]])
                else: # if there are not enough, we sample with replacement (some samples will be repeated)
                    permuted.append(vals_add[torch.randint(0, num_add, (num,))])

        perm = torch.cat(permuted)
        return perm[torch.argsort(sorted_indices)].tolist() # b => sorted_b' = sorted_a <= a