import os
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

# importing required modules
from zipfile import ZipFile

# from src.datamodules.components.diagvib_dataset import TorchDatasetWrapper_env,DiagVibSixDatasetSimple
from src.data.components.diagvib_dataset import DiagVibSixDatasetSimple


class DiagvibDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir,
        batch_size,
        num_workers,
        pin_memory,
    ):
        super().__init__()

        self.dataset_dir = os.path.join(data_dir, "dataset")

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # self.train_transform = None
        # self.val_transform = None
        self.metadata_fields = [
            "shape",
            "hue",
            "lightness",
            "texture",
            "position_factor",
            "scale_factor",
        ]

    def prepare_data(self):
        # specifying the zip file name
        print("diagvib dataset dir path:", self.dataset_dir)
        if os.path.exists(self.dataset_dir) is False:
            file_zip = os.path.join(self.root_dir, "dataset.zip")

            with ZipFile(file_zip, "r") as zip:
                # printing all the contents of the zip file
                # 	zip.printdir()

                # extracting all the files
                print("Extracting all the files now...")
                zip.extractall(self.root_dir)
                print("Done!")

    def setup(self, stage=None):
        # called on every GPU
        if stage == "fit" or stage is None:
            path_train = os.path.join(self.dataset_dir, "train")
            self.train_ds = DiagVibSixDatasetSimple(root_dir=path_train)

            path_validation = os.path.join(self.dataset_dir, "validation")

            self.val_ds = DiagVibSixDatasetSimple(root_dir=path_validation)

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=True,
            num_workers=self.num_workers,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=True,
            num_workers=self.num_workers,
        )
        return val_loader
