"""
This test will check that the data passed to the model is the same in the CPU and DDP configurations.
"""

import hydra
from omegaconf import DictConfig
from typing import Optional

import torch
from torch.utils.data.sampler import RandomSampler
from torch.utils.data import DataLoader, DistributedSampler
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.trainer.supporters import CombinedLoader

from .utils import plot_images_multienv

class TestingModule(LightningModule):
    def __init__(
        self,
        classifier: torch.nn.Module,
        expected_results: DictConfig,
        ):
        super().__init__()

        # Retrieve classifier to deduce the logits
        self.model = classifier
        first_param = True
        for param in self.model.parameters():
            param.requires_grad = False
            if first_param:
                param.requires_grad = True
                first_param = False

        self.loss = torch.nn.CrossEntropyLoss()

        # Save expected results as a DictConfig to be used in the testing step
        self.expected_results = expected_results
        self.size_main, self.size_pa, self.size_palogits = 0, 0, 0
        self.plot_images = None
    
    def _testing_step(self, batch_dict: dict, bidx: int):
        b_main, b_pa, b_palogits = batch_dict["dl_main"], batch_dict["dl_pa"], batch_dict["dl_palogits"]
        env_names = list(b_main.keys())

        if bidx == 0 and self.trainer.local_rank == 0:
            # Save some images to plot later and have a visual inspection.
            self.plot_images = [b_main[env][0][0] for env in env_names]

        """
        We will use the model only for one epoch, so we are interested in checking the size of the data being passed
        wrt the same procedure in DDP.
        """
        self.size_main += len(b_main[env_names[0]][1])
        self.size_pa += len(b_pa[env_names[0]][1])
        self.size_palogits += len(b_palogits[env_names[0]][1])
        # In the configuration of the test, all dataloaders must have the same data source.
        # Eventually some samples will be discarded for PA and PA_logits wrt the main dataloader, but this shoudln't be
        # reflected here, as mode="min_size" is used in the CombinedLoader.
        assert self.size_main == self.size_pa == self.size_palogits, "The batches must have the same length."

        """
        Observations must be shuffled by the dataloaders in different ways.
        """ 

        labels_main = torch.cat([b_main[env][1] for env in env_names])
        labels_pa = torch.cat([b_pa[env][1] for env in env_names])
        labels_palogits = torch.cat([b_palogits[env][1] for env in env_names])
        assert not torch.equal(labels_main, labels_pa), "Observations in the PA dataloader are not being shuffled properly."
        assert not torch.equal(labels_pa, labels_palogits), "Observations in the PA_logits dataloader are not being shuffled properly."

        plot_images_multienv
        
        for env in env_names:
            if env != env_names[0]:
                """
                Observations across environments must be different when shift_factor > 0.
                Use large shift_factor to ensure that the samples are not repeated and it's a false positive.
                """
                assert not torch.equal(Xe_main, b_main[env][0]), "The samples across environments could be repeated in the main dataset."
                assert not torch.equal(Xe_pa, b_pa[env][0]), "The samples across environments could be repeated in the PA dataset."
                assert not torch.equal(Xe_palogits, b_palogits[env][0]), "The samples across environments could be repeated in the PA_logits dataset."

                """
                Labels across environments must be the same in any case, at least for the PA and PA_logits dataloaders.
                """
                if self.expected_results.main.corresponding_labels:
                    assert torch.equal(ye_main, b_main[env][1]), "Labels in the main dataloader are not corresponding, and they should."
                else:
                    assert not torch.equal(ye_main, b_main[env][1]), "Labels in the main dataloader are corresponding, and they shouldn't be."
                assert torch.equal(ye_pa, b_pa[env][1]), "Labels in the PA dataloader are not corresponding, and they should be."
                assert torch.equal(ye_palogits, b_palogits[env][1]), "Labels in the PA_logits dataloader are not corresponding, and they should be."

            Xe_main, Xe_pa, Xe_palogits = b_main[env][0], b_pa[env][0], b_palogits[env][0]
            ye_main, ye_pa, ye_palogits = b_main[env][1], b_pa[env][1], b_palogits[env][1]


    def _model_step(self, batch_dict: dict, grad_enabled: bool = True):
        b_main = batch_dict["dl_main"]
        env_names = list(b_main.keys())
        x, y = b_main[env_names[0]]
        with torch.set_grad_enabled(grad_enabled):
            logits = self.model(x).to(self.device)
            loss = self.loss(input=logits, target=y).to(self.device)

        return loss
    
    def training_step(self, batch_dict: dict, bidx: int):
        if self.trainer.accelerator == "gpu":
            print(f"This is happening: {bidx}")
            print(self.trainer.local_rank)
        self._testing_step(batch_dict, bidx)

        loss = self._model_step(batch_dict)
        return {"loss": loss}

    def validation_step(self, batch_dict: dict, bidx: int):
        self._testing_step(batch_dict, bidx)
        loss = self._model_step(batch_dict, False)
        return {"loss": loss}
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.1)
        return {"optimizer": optimizer}


class TestingDataModule(LightningDataModule):
    def __init__(
        self,
        cfg: DictConfig,
    ):
        super().__init__()

        self.dm_main = hydra.utils.instantiate(cfg.ddp.datamodules.main) 
        self.dm_pa = hydra.utils.instantiate(cfg.ddp.datamodules.pa)
        self.dm_palogits = hydra.utils.instantiate(cfg.ddp.datamodules.pa_logits)


    def prepare_data(self):
        self.dm_main.prepare_data()
        self.dm_pa.prepare_data()
        self.dm_palogits.prepare_data()

    def setup(self, stage: Optional[str] = None):
        self.dm_main.setup("fit")
        self.dm_pa.setup("fit")
        self.dm_palogits.setup("fit")

        self.ds_pa = self.dm_pa.train_ds
        self.ds_palogits = self.dm_palogits.logits_ds

    def train_dataloader(self):
        return CombinedLoader(
            {
                "dl_main": self.dm_main.train_dataloader(), 
                "dl_pa": self.dm_pa.train_dataloader(),
                "dl_palogits": self.dm_palogits.train_dataloader()
            },
            mode="min_size"
        )

    def val_dataloader(self):
        return CombinedLoader(
            {
                "dl_main": self.dm_main.val_dataloader(), 
                "dl_pa": self.dm_pa.val_dataloader(),
                "dl_palogits": self.dm_palogits.val_dataloader()
            },
            mode="min_size"
        )


def test_ddp(cfg: DictConfig):
    """
    The goal is to evaluate the data retrieved by the Trainer when using DDP.
    """

    """
    We run the TestingModule with both CPU and DDP trainers.
    """
    # If the DataModule requires a classifier, we will assume it's the same as the one used in the logits
    # otherwise the results wouldn't make any sense.
    model = TestingModule(
        classifier=hydra.utils.instantiate(cfg.ddp.datamodules.pa_logits.classifier),
        expected_results=cfg.ddp.expected_results
    )

    # We initialize the datamodule yielding a CombinedLoader
    dm = TestingDataModule(cfg)
    dm.prepare_data()
    dm.setup()

    trainer = hydra.utils.instantiate(cfg.ddp.trainer.cpu)
    trainer.fit(model, datamodule=dm)
    size_main, size_pa, size_palogits = model.size_main, model.size_pa, model.size_palogits # store the sizes
    devices_cpu = trainer.device_ids

    plot_images_multienv(
        model.plot_images,
        [str(i) for i in range(len(model.plot_images))],
        cfg.paths.results_tests + "/cpu"
    )

    model = TestingModule(
        classifier=hydra.utils.instantiate(cfg.ddp.datamodules.pa_logits.classifier),
        expected_results=cfg.ddp.expected_results
    )

    trainer = hydra.utils.instantiate(cfg.ddp.trainer.ddp)
    trainer.fit(model, datamodule=dm)
    devices_ddp = trainer.device_ids

    print("\nDevices used by CPU: ", devices_cpu)
    print("Devices used by DDP: ", devices_ddp)


    plot_images_multienv(
        model.plot_images,
        [str(i) for i in range(len(model.plot_images))],
        cfg.paths.results_tests + "/ddp"
    )


    """
    The size of the data passed through the model should be the same in CPU and DDP configurations.
    """
    print("HERE IS WHERE THE CONFLICT ARISES", size_main, model.size_main)
    assert size_main == model.size_main, "The size of the main dataloader is different when using DDP."
    assert size_pa == model.size_pa, "The size of the PA dataloader is different when using DDP."
    assert size_palogits == model.size_palogits, "The size of the PA_logits dataloader is different when using DDP."
    print("\nSize CPU vs DDP: ")
    print("Main: ", size_main, model.size_main)
    print("PA: ", size_pa, model.size_pa)
    print("PA_logits: ", size_palogits, model.size_palogits)

    """
    The size of the data passed through the model in the PA and PA_logits case should be equal than the size of the dataset.
    No possible drop_last=True because the adjustment has already been made.
    """
    assert size_pa == len(dm.ds_pa), "The size of the PA dataloader is different than the size of the dataset. Check drop_last=False."
    assert size_palogits == len(dm.ds_palogits), "The size of the PA_logits dataloader is different than the size of its dataset. Check drop_last=False."
    assert size_pa == size_palogits, "The size of the PA dataset is different than the size of the PA_logits dataset."

    print("\n\nTest passed.")