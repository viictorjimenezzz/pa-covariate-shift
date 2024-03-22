"""
This test will check that the PA module provides the same results using PA and PA_logits, for both CPU and DDP configurations.
"""

import hydra
from omegaconf import DictConfig

import torch
from torch.utils.data import DataLoader, SequentialSampler
from pytorch_lightning import LightningModule, LightningDataModule

import os
import pandas as pd

from .utils import get_acc_metrics, get_pa_metrics

def test_pa_module(cfg: DictConfig):
    AFR_pred, AFR_true, acc_pa = [], [], [] # store for comparison at the end

    # ________________________________________ CPU ________________________________________
    # PA DataModule -----------------------------------------------------------------------
    datamodule_pa: LightningDataModule = hydra.utils.instantiate(cfg.pa_module.datamodules.pa)
    datamodule_pa.prepare_data()
    datamodule_pa.setup("fit")

    # We will use the same datalaoder but with a SequentialSampler, so that results are the same in all cases
    pa_traindl = datamodule_pa.train_dataloader()
    pa_traindl = DataLoader(
        pa_traindl.dataset, 
        batch_size=pa_traindl.batch_size, 
        sampler=SequentialSampler(pa_traindl.dataset), 
        num_workers=pa_traindl.num_workers, 
        collate_fn=pa_traindl.collate_fn
    )

    # They are the same dataloader, but still wanna check that everything is fine
    pa_valdl = datamodule_pa.val_dataloader()
    pa_valdl = DataLoader(
        pa_valdl.dataset, 
        batch_size=pa_valdl.batch_size, 
        sampler=SequentialSampler(pa_valdl.dataset), 
        num_workers=pa_valdl.num_workers, 
        collate_fn=pa_valdl.collate_fn
    )

    # Training the model:
    pamodule_pa: LightningModule = hydra.utils.instantiate(cfg.pa_module.pa_lightningmodule.pa)
    trainer = hydra.utils.instantiate(cfg.pa_module.trainer.cpu)
    trainer.fit(
        model=pamodule_pa, 
        train_dataloaders=pa_traindl,
        val_dataloaders=pa_valdl
    )
    acc_metrics = get_acc_metrics(pamodule_pa)
    AFR_pred.append(acc_metrics[0])
    AFR_true.append(acc_metrics[1])
    acc_pa.append(acc_metrics[2])

    beta_epoch_pa, logPA_epoch_pa = get_pa_metrics(pamodule_pa)
    assert len(beta_epoch_pa) == cfg.pa_module.trainer.cpu.max_epochs, "Some beta values are not being stored properly."
    assert len(logPA_epoch_pa) == cfg.pa_module.trainer.cpu.max_epochs, "Some logPA values are not being stored properly."

    # PA_logits DataModule ------------------------------------------------------------------
    datamodule_palogs: LightningDataModule = hydra.utils.instantiate(cfg.pa_module.datamodules.pa_logits)
    datamodule_palogs.prepare_data()
    datamodule_palogs.setup("fit")

    palogs_traindl = datamodule_palogs.train_dataloader()
    palogs_traindl = DataLoader(
        palogs_traindl.dataset, 
        batch_size=palogs_traindl.batch_size, 
        sampler=SequentialSampler(palogs_traindl.dataset), 
        num_workers=palogs_traindl.num_workers, 
        collate_fn=palogs_traindl.collate_fn
    )

    # They are the same dataloader, but still wanna check that everything is fine
    palogs_valdl = datamodule_palogs.val_dataloader()
    palogs_valdl = DataLoader(
        palogs_valdl.dataset, 
        batch_size=palogs_valdl.batch_size, 
        sampler=SequentialSampler(palogs_valdl.dataset), 
        num_workers=palogs_valdl.num_workers, 
        collate_fn=palogs_valdl.collate_fn
    )

    # Training the model:
    pamodule_palogs: LightningModule = hydra.utils.instantiate(cfg.pa_module.pa_lightningmodule.pa_logits)
    trainer = hydra.utils.instantiate(cfg.pa_module.trainer.cpu)
    trainer.fit(
            model=pamodule_palogs, 
            train_dataloaders=palogs_traindl, 
            val_dataloaders=palogs_valdl
    )
    acc_metrics = get_acc_metrics(pamodule_palogs)
    AFR_pred.append(acc_metrics[0])
    AFR_true.append(acc_metrics[1])
    acc_pa.append(acc_metrics[2])

    beta_epoch_palogs, logPA_epoch_palogs = get_pa_metrics(pamodule_palogs)
    assert len(beta_epoch_palogs) == cfg.pa_module.trainer.cpu.max_epochs, "Some beta values are not being stored properly."
    assert len(logPA_epoch_palogs) == cfg.pa_module.trainer.cpu.max_epochs, "Some logPA values are not being stored properly."


    """
    The results from PA and PA_logits implementations should be the same.
    """
    assert torch.equal(beta_epoch_pa, beta_epoch_palogs), "The beta values are not equal between PA and PA_logits methods."
    assert torch.equal(logPA_epoch_pa, logPA_epoch_palogs), "The logPA values are not equal between PA and PA_logits methods."


    print("\nCPU tests passed.\n")
    # ________________________________________ DDP ________________________________________
    # Now I don't need to initialize so many things.

    # PA model ----------------------------------------------------------------------------
    pamodule_pa: LightningModule = hydra.utils.instantiate(cfg.pa_module.pa_lightningmodule.pa)
    trainer = hydra.utils.instantiate(cfg.pa_module.trainer.ddp) # DDP trainer
    trainer.fit(
        model=pamodule_pa, 
        train_dataloaders=pa_traindl,
        val_dataloaders=pa_valdl
    )
    acc_metrics = get_acc_metrics(pamodule_pa)
    AFR_pred.append(acc_metrics[0])
    AFR_true.append(acc_metrics[1])
    acc_pa.append(acc_metrics[2])

    beta_epoch_pa, logPA_epoch_pa = get_pa_metrics(pamodule_pa)
    assert len(beta_epoch_pa) == cfg.pa_module.trainer.ddp.max_epochs, "Some beta values are not being stored properly."
    assert len(logPA_epoch_pa) == cfg.pa_module.trainer.ddp.max_epochs, "Some logPA values are not being stored properly."

    """
    The results in DDP should be the same as in CPU. We compare with the last ones saved (PA_logits).
    """
    # We use more epochs on DDP, and thus we compare the first cpu.max_epochs epochs.

    assert torch.equal(beta_epoch_pa[:cfg.pa_module.trainer.cpu.max_epochs], beta_epoch_palogs), "The beta values are not equal between CPU and DDP implementations."
    assert torch.equal(logPA_epoch_pa[:cfg.pa_module.trainer.cpu.max_epochs], logPA_epoch_palogs), "The logPA values are not equal between CPU and DDP implementations."


    # PA_logits model -------------------------------------------------------------------------
    pamodule_palogs: LightningModule = hydra.utils.instantiate(cfg.pa_module.pa_lightningmodule.pa_logits)
    trainer = hydra.utils.instantiate(cfg.pa_module.trainer.ddp) # DDP trainer
    trainer.fit(
        model=pamodule_palogs, 
        train_dataloaders=palogs_traindl,
        val_dataloaders=palogs_valdl
    )
    acc_metrics = get_acc_metrics(pamodule_palogs)
    AFR_pred.append(acc_metrics[0])
    AFR_true.append(acc_metrics[1])
    acc_pa.append(acc_metrics[2])

    beta_epoch_palogs, logPA_epoch_palogs = get_pa_metrics(pamodule_palogs)
    assert len(beta_epoch_palogs) == cfg.pa_module.trainer.ddp.max_epochs, "Some beta values are not being stored properly."
    assert len(logPA_epoch_palogs) == cfg.pa_module.trainer.ddp.max_epochs, "Some logPA values are not being stored properly."

    """
    The results for PA and PA_logits should be the same also in DDP setting.
    The comparison with CPU has already been done.
    """
    # We use more epochs on DDP, and thus we compare the first cpu.max_epochs epochs.
    assert torch.equal(beta_epoch_pa, beta_epoch_palogs), "The beta values are not equal between CPU and DDP implementations."
    assert torch.equal(logPA_epoch_pa, logPA_epoch_palogs), "The logPA values are not equal between CPU and DDP implementations."
    
    # ______________________________________________________________________________________

    """
    Finally we compare the results of the AFR and acc_pa metrics in the 4 settings.
    """
    assert AFR_pred[0] == AFR_pred[1] == AFR_pred[2] == AFR_pred[3], "The AFR_pred values are not equal."
    assert AFR_true[0] == AFR_true[1] == AFR_true[2] == AFR_true[3], "The AFR_true values are not equal."
    assert acc_pa[0] == acc_pa[1] == acc_pa[2] == acc_pa[3], "The acc_pa values are not equal."