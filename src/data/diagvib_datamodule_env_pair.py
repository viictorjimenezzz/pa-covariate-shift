import os
from torch.utils.data import Dataset, DataLoader, Subset
from pytorch_lightning import LightningDataModule
from torchvision import transforms as transform_lib
import torchvision.transforms.functional as F
from torch.utils.data import ConcatDataset
# importing required modules
from zipfile import ZipFile
from torch import triu, eq,tensor

# from src.datamodules.components.diagvib_dataset import TorchDatasetWrapper_env,DiagVibSixDatasetSimple
from src.datamodules.components.diagvib_dataset import DiagVibSixDatasetSimple

class DiagvibDatamodule(LightningDataModule):

    def __init__(self,
                 root_dir,
                 batch_size,
                 num_workers,
                 pin_memory,
                 grouping_fields=None
                 ):

        super().__init__()

        self.root_dir = root_dir
        self.dataset_dir = os.path.join(root_dir,'dataset')

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.grouping_fields=grouping_fields


        # self.train_transform = None
        # self.val_transform = None
        self.metadata_fields = ['shape', 'hue', 'lightness', 'texture', 'position_factor', 'scale_factor']

    def prepare_data(self):
        # specifying the zip file name
        if os.path.exists(self.dataset_dir) is False:
            file_zip = os.path.join(self.root_dir, 'dataset.zip')

            with ZipFile(file_zip, 'r') as zip:
                # printing all the contents of the zip file
                # 	zip.printdir()

                # extracting all the files
                print('Extracting all the files now...')
                zip.extractall(self.root_dir)
                print('Done!')




    def setup(self, stage=None):

        # called on every GPU
        if stage == "fit" or stage is None:
            self.path_train_envs = [os.path.join(self.dataset_dir, path) for path in os.listdir(self.dataset_dir) if
                                    'train' in path]

            self.path_val = os.path.join(self.dataset_dir, 'valod')

            train_envs_ds = []
            for path_train_env in self.path_train_envs:
                train_envs_ds.append(DiagVibSixDatasetSimple(root_dir=path_train_env))
            # print(train_envs_ds)
            self.train_ds = ConcatDataset(train_envs_ds)
            # self.train_ds = train_envs_ds[0]
            self.val_ds = DiagVibSixDatasetSimple(root_dir=self.path_val)

        if self.grouping_fields:
            self.group_index = [self.metadata_fields.index(grouping_field) \
                                for grouping_field in self.grouping_fields]
            # self.grouper = CombinatorialGrouper(self.dataset, self.grouping_fields)

        if stage == "test" or stage is None:

            self.path_test_od = os.path.join(self.dataset_dir, 'test')
            self.path_test_id = os.path.join(self.dataset_dir, 'valid')
            self.test_od_ds = DiagVibSixDatasetSimple(root_dir=self.path_test_od)
            self.test_id_ds = DiagVibSixDatasetSimple(root_dir=self.path_test_id)


    def train_dataloader(self):

        train_loader = DataLoader(
            self.train_ds,
            batch_size= self.batch_size,
            pin_memory = True,
            shuffle = True,
            num_workers = self.num_workers,
        )
        return train_loader


    def val_dataloader(self):

        val_loader = DataLoader(
            self.val_ds,
            batch_size= self.batch_size,
            pin_memory = True,
            shuffle = True,
            num_workers = self.num_workers,
        )
        return val_loader


    def test_dataloader(self):

        test_loader_id = DataLoader(
            self.test_id_ds,
            batch_size= self.batch_size,
            pin_memory = True,
            shuffle = True,
            num_workers = self.num_workers,
        )

        test_loader_od = DataLoader(
            self.test_od_ds,
            batch_size= self.batch_size,
            pin_memory = True,
            shuffle = True,
            num_workers = self.num_workers,
        )

        return  [test_loader_id, test_loader_od]
        # return  test_loader_od

    def eval(self, preds, y , meta):
        return None
            # metrics_wilds,_ = self.dataset.eval(preds,y,meta)
            # metrics_chosen = ['acc_avg','acc_wg']
            #
            # metrics_wilds = {str(metric):val for metric,val in metrics_wilds.items() if metric in metrics_chosen}
            #
            # # print(acc)

        # return metrics_wilds

    def create_mask_ys_envs(self, y, meta_env):

        # Grouping
        group_index = self.group_index

        mask_y = y.contiguous().view(-1, 1)
        mask_y = triu(eq(mask_y, mask_y.T), diagonal=1)

        # IMPORTANT - in case of more than one variable for the environment it is possible to do XOR; currently only doing and for more than 1 variable defining the environment
        mask_meta_env_pre = eq(meta_env[:, group_index].repeat((meta_env.shape[0], 1, 1)),
                               meta_env[:, group_index].repeat((meta_env.shape[0], 1, 1)).swapaxes(0, 1)).sum(dim=2)
        mask_meta_env = mask_meta_env_pre.detach().clone()
        mask_meta_env[mask_meta_env_pre != 0] = False
        mask_meta_env[mask_meta_env_pre == 0] = True
        mask_y_meta_env = mask_y * mask_meta_env

        return mask_y_meta_env