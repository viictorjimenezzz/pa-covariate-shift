"""
This test will check that the data generated in the MultiEnv, PA and PA_logits datamodules is consistent with the
requirements of the different optimization procedures.
"""

import hydra
from omegaconf import DictConfig

import os
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import warnings

from src.pa_metric.pairing import PosteriorAgreementDatasetPairing
from pytorch_lightning import LightningDataModule

from src.data.components import MultienvDataset, LogitsDataset
from src.data.components.collate_functions import MultiEnv_collate_fn

from .utils import plot_multienv, plot_images_multienv

__all__ = ["test_dataloaders", "test_sampler"]
PLOT_INDS = [0, 9] # must be smaller than batch_size
CHECK_LOGITS = 10 # will be passed to the model, must be smaller than the length of the paired dataset

"""
- These tests are designed to run locally, as subprocess tests are performed separately. Reduce CHECK_LOGITS and batch_size if you run out of memory.
- It is recommended to keep batch_size and CHECK_LOGITS as big as possible, otherwise randomness could lead to some false negatives.
- Important to keeep batch_size <= len(dataset)/2, as I run two epochs here.
"""

def _dl_plot(dataloader: DataLoader, batch_size: int, img_dir: str, expected: str = "equal"):
    assert expected in ["equal", "not_equal"], "The expected behaviour must be either 'equal' or 'not_equal'."

    for epoch in range(2):
        for batch_ind, batch in enumerate(dataloader):
            env_names = list(batch.keys())
            x1, x2, y1, y2 = batch[env_names[0]][0], batch[env_names[1]][0], batch[env_names[0]][1], batch[env_names[1]][1]
            
            """
            Labels will be the same when the dataset is correspondent. PA and PA_logits must be, but main can be different.
            """
            if expected == "equal":
                assert torch.equal(y1, y2), "The labels are not the same, and they should."
            else:
                assert not torch.equal(y1, y2), "The labels are the same, and they shouldn't."

            assert torch.equal(torch.tensor(x1.size()), torch.tensor(x2.size())), "The images have different sizes."
            assert x1.size(0) == batch_size, "The batch size is not the same as the one specified in the config file."

            # Visual check:
            for ind in PLOT_INDS:
                plot_images_multienv([x1[ind], x2[ind]], [y1[ind], y2[ind]],
                                     os.path.join(img_dir, f"epoch_{epoch}_batch_{batch_ind}_ind_{ind}"))
                
            break # only one batch per epoch

def _dl_logits(dataloader: DataLoader, classifier: torch.nn.Module, logits_dataloader: DataLoader, expected="equal"):
    assert expected in ["equal", "not_equal"], "The expected behaviour must be either 'equal' or 'not_equal'."

    assert len(dataloader.dataset) == len(logits_dataloader.dataset), "The number of samples in the dataset and the logits dataset is not the same."
    #inds_to_compare = torch.randint(0, len(dataloader.dataset), (CHECK_LOGITS,)).tolist()
    inds_to_compare = list(range(5))
    subset = dataloader.dataset.__getitems__(inds_to_compare)
    subset_logits = logits_dataloader.dataset.__getitems__(inds_to_compare)

    classifier.eval()
    with torch.no_grad():
        if expected == "equal":
            for i in range(2): # two environments
                assert torch.allclose(classifier(subset[i][0]), subset_logits[i][0]), "The logits generated and stored are not the same, and they should be."
        else:
            for i in range(2):
                assert not torch.allclose(classifier(subset[i][0]), subset_logits[i][0]), "The logits generated and stored are the same, and they shouldn't be."

def test_dataloaders(cfg: DictConfig):
    """
    EXPLANATION
    """
    torch.manual_seed(cfg.seed) # so that the dataloader shuffle yields the same results

    # Main_dataloader
    dm_main: LightningDataModule = hydra.utils.instantiate(cfg.data.datamodules.main) 
    dm_main.prepare_data()
    dm_main.setup("fit")
    dl_main = dm_main.train_dataloader()
    assert type(dl_main.dataset) in [MultienvDataset, LogitsDataset], "All datasets must belong to class MultienvDataset or LogitsDataset."

    _dl_plot(dl_main, 
             cfg.data.datamodules.main.batch_size, 
             os.path.join(cfg.paths.results_tests, cfg.data.datamodules.data_name + "_main"),
             expected = "equal" if cfg.data.expected_results.main.corresponding_labels else "not_equal")
    
    """
    We must check whether the main dataset contains corresponding labels, which usually is not the case.
    """
    if cfg.data.expected_results.main.corresponding_labels:
        for e in range(1, dm_main.train_ds.num_envs): # Main dataset can have more than two environments
            assert torch.equal(dm_main.train_ds.__getlabels__(list(range(len(dm_main.train_ds))))[0],
                               dm_main.train_ds.__getlabels__(list(range(len(dm_main.train_ds))))[e]), "The labels in the main dataset are not corresponding, and they should be."
    else:
        for e in range(1, dm_main.train_ds.num_envs):
            assert not torch.equal(dm_main.train_ds.__getlabels__(list(range(len(dm_main.train_ds))))[0],
                                   dm_main.train_ds.__getlabels__(list(range(len(dm_main.train_ds))))[e]), "The labels in the main dataset are corresponding, and they shouldn't be."

    
    # PA_dataloader
    dm_pa: LightningDataModule = hydra.utils.instantiate(cfg.data.datamodules.pa)
    dm_pa.prepare_data()
    dm_pa.setup("fit")
    dl_pa = dm_pa.train_dataloader()
    assert type(dl_pa.dataset) in [MultienvDataset, LogitsDataset], "All datasets must belong to class MultienvDataset or LogitsDataset."

    _dl_plot(dl_pa, 
             cfg.data.datamodules.pa.batch_size, 
             os.path.join(cfg.paths.results_tests, cfg.data.datamodules.data_name + "_pa"),
             expected = "equal") # this must be always the case for PA
    
    """
    We must check that the PA dataset contains corresponding labels, as it has been passed through PosteriorAgreementDatasetPairing
    """
    assert torch.equal(dm_pa.train_ds.__getlabels__(list(range(len(dm_pa.train_ds))))[0],
                       dm_pa.train_ds.__getlabels__(list(range(len(dm_pa.train_ds))))[1]), "The labels in the PA dataset are not corresponding."

    # PAlogits_dataloader
    dm_palogits: LightningDataModule = hydra.utils.instantiate(cfg.data.datamodules.pa_logits)
    dm_palogits.prepare_data()
    dm_palogits.setup("fit")
    dl_palogits = dm_palogits.train_dataloader() # Has corresponding labels by definition
    assert type(dl_palogits.dataset) in [MultienvDataset, LogitsDataset], "All datasets must belong to class MultienvDataset or LogitsDataset."

    """
    We must check that the labels and logits are the same when `shuffle=False`, for CHECK_LOGITS_INDS. 
    Set `expected="equal"` when the model passed to the PA_logits dataloader is the same as the one passed to the main and PA dataloaders (if they require so).
    """
    if "classifier" in list(cfg.data.datamodules.main.keys()): # If the main DL has a classifier (only in adversarial case)
        model_main: torch.nn.Module = hydra.utils.instantiate(cfg.data.datamodules.main.classifier)
        _dl_logits(dl_main, model_main, dl_palogits,
                   # Expected equal when labels are correspondent (so PAPairing won't affect it) and they have the same input model (so the logits are the same)
                   expected="equal" if cfg.data.expected_results.main.corresponding_labels and cfg.data.expected_results.main.same_model_logits else "not_equal")
        
        model_pa: torch.nn.Module = hydra.utils.instantiate(cfg.data.datamodules.pa.classifier)
        _dl_logits(dl_pa, model_pa, dl_palogits,
                   expected="equal" if cfg.data.expected_results.pa.same_model_logits else "not_equal") # Must have corresponding labels

    model_palogits: torch.nn.Module = hydra.utils.instantiate(cfg.data.datamodules.pa_logits.classifier)
    # _dl_logits(dl_main, model_palogits, dl_palogits, 
    #            expected="equal" if cfg.data.expected_results.main.corresponding_labels else "not_equal") # so pairing won't affect it
    _dl_logits(dl_pa, model_palogits, dl_palogits, expected="equal") # dl_pa paired, and pairing is maintained in dl_palogits

    """
    Check length of the datasets.
    """
    assert len(dm_main.train_ds) == len(dm_pa.train_ds) == len(dm_palogits.logits_ds), "The length of the datasets is not the same."

    """
    Finally, the PA and PA_logits datasets should have the same labels. Checking for the first environment is enough.
    """
    assert torch.equal(dm_pa.train_ds.__getlabels__(list(range(len(dm_pa.train_ds))))[0],
                       dm_palogits.logits_ds.y), "The labels in the PA and PA_logits datasets are not the same."    

    # print("\n Dataloader retrieval (first 5 samples or first environment): ")
    """
    To compare the dataloader retrieval, we should see that the samples of the main and PA dataloaders are not the same.
    Only the first batch is enough
    """
    for bidx, (b_main, b_pa, b_palog) in enumerate(zip(dl_main, dl_pa, dl_palogits)):
        env_names = list(b_main.keys())
        for env in env_names:
            Xe_main, Xe_pa, Xe_palog = b_main[env][0], b_pa[env][0], b_palog[env][0]
            ye_main, ye_pa, ye_palog = b_main[env][1], b_pa[env][1], b_palog[env][1]

            sum_main = torch.tensor([torch.sum(X).item() for X in Xe_main])
            sum_pa = torch.tensor([torch.sum(X).item() for X in Xe_pa])
            assert not torch.equal(sum_main, sum_pa), "The PA dataloader doesn't shuffle observations properly."
        break # only the first batch

    print("\n\nTest passed.")

    
def test_sampler(cfg: DictConfig):
    """
    The goal is to see whether the PosteriorAgreementDatasetPairing function works as expected.
        1. Compare observations from the dataset given by the original permutations and the ones given by the sampler.
        2. Analyze observations provided by the train_dataloader for different epochs.
    """

    np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    #warnings.simplefilter("ignore") # generation of images will yield warning

    dm: LightningDataModule = hydra.utils.instantiate(cfg.data.datamodule)
    dm.prepare_data()
    dm.setup("fit")

    # In this way we can keep both datasets in memory:
    dataset = MultienvDataset(dm.train_ds.dset_list)
    dataset_sampler = PosteriorAgreementDatasetPairing(dm.train_ds)
    
    print("Class of dataset: ", type(dataset), "\n")
    print("Length original vs sampled: {} vs {}".format(len(dataset), len(dataset_sampler)))

    envs, envs_sampled = list(dataset.__getitem__(0).keys()), list(dataset.__getitem__(0).keys())
    print("Number of environments original vs sampled: {} vs {}".format(len(dataset), len(dataset_sampler)))
    assert len(envs) == len(envs_sampled), "The number of environments is not the same."

    # Since datasets are MiltienvDataset objects, we can get their labels straight away:
    print("\nOriginal labels: ", )
    for e in range(len(envs)):
        print("Environment " + str(envs[e]) + ": ", list(dataset.__getlabels__(list(range(len(dataset)))))[e].tolist()[:10])
    print("\nSampled labels: ", )
    for e in range(len(envs_sampled)):
        print("Environment " + str(envs_sampled[e]) + ": ", list(dataset_sampler.__getlabels__(list(range(len(dataset_sampler)))))[e].tolist()[:10])

    #inds_to_plot = [0, 11, 10, 3] # 0-0, 0-1, 1-0, 1-1
    inds_to_plot = [1, 2, 4, 5] # 8, 8, 6, 6
    plot_multienv(dataset, cfg.paths.results_tests + "dataset/test_dataset", random=inds_to_plot)
    plot_multienv(dataset_sampler, cfg.paths.results_tests + "dataset/test_sampler", random=inds_to_plot)

    # Now I will generate a mismatch of 100 samples between the two environments in the original dataset:
    subset_inds = torch.randint(0, len(dataset), (20,))
    envs_subset = dataset.__getitems__(subset_inds) 

    print("\nBefore modification: ")
    print("Subset env 0: ", envs_subset[0][1])
    print("Subset env 1: ", envs_subset[1][1])

    inds_1 = torch.where(envs_subset[1][1].eq(envs_subset[1][1][0]))[0] # position of observations in env2 of label equal to first label
    inds_0 = torch.where(envs_subset[0][1].ne(envs_subset[1][1][0]))[0] # position of observations in env1 of label different to such label
    envs_subset[0][0][inds_0[0:len(inds_1)]] = envs_subset[1][0][inds_1] # change observations
    envs_subset[0][1][inds_0[0:len(inds_1)]] = envs_subset[1][1][inds_1] # change labels associated

    print("\nAfter modification: ")
    print("Subset env 0: ", envs_subset[0][1])
    print("Subset env 1: ", envs_subset[1][1])

    subset = MultienvDataset([TensorDataset(*subs_env) for subs_env in envs_subset]) # create subset

    print("\nSubset labels for modified indexes: ", inds_0[0:len(inds_1)].tolist())
    envs = list(subset.__getitem__(0).keys())
    for e in range(len(envs)):
        print("Environment " + str(envs[e]) + ": ", list(subset.__getlabels__(list(range(len(subset)))))[e].tolist())

    subset_sampler = PosteriorAgreementDatasetPairing(MultienvDataset(subset.dset_list))

    print("\nSampled labels for modified: ", )
    envs_sampled = list(subset_sampler.__getitem__(0).keys())
    for e in range(len(envs_sampled)):
        print("Environment " + str(envs_sampled[e]) + ": ", list(subset_sampler.__getlabels__(list(range(len(subset_sampler)))))[e].tolist())

    print("\nLength original vs sampled: {} vs {}".format(len(subset), len(subset_sampler)))
    print("Number of environments original vs sampled: {} vs {}".format(len(envs), len(envs_sampled)))

    inds_to_plot = [1, 2, 4, 5] # 8, 8, 6, 6
    plot_multienv(subset, cfg.paths.results_tests + "subset/test_subset", random=inds_to_plot)
    plot_multienv(subset_sampler, cfg.paths.results_tests + "subset/test_subsetsampler", random=inds_to_plot)

    print("\n\nTest passed.")