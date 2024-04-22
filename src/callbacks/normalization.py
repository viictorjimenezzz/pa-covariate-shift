from typing import Union, Callable
from pytorch_lightning.callbacks import Callback
from torchvision.transforms import Normalize

class Normalization_Callback(Callback):
    """
    Normalizes the batch before feeding it into the model. The attributes <phase>.mean and <phase>.std are expected to be
    present in the LightningDataModule, where <phase> = {train, val, test}.
    """

    @staticmethod
    def _normalize_batch(batch: Union[tuple, dict], normalize_fn: Callable):
        if isinstance(batch, dict):
            for env in batch.keys():
                batch[env] = (normalize_fn(batch[env][0]), batch[env][1])
        else:
            batch = (normalize_fn(batch[0]), batch[1])

    def on_train_batch_start(self, trainer, pl_module, batch: Union[tuple, dict], batch_idx):
        normalize = Normalize(
            mean=trainer.datamodule.train_mean,
            std=trainer.datamodule.train_std
        )
        self._normalize_batch(batch, normalize)
        
    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        normalize = Normalize(
            mean=trainer.datamodule.val_mean,
            std=trainer.datamodule.val_std
        )
        self._normalize_batch(batch, normalize)

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        normalize = Normalize(
            mean=trainer.datamodule.test_mean,
            std=trainer.datamodule.test_std
        )
        batch[0] = (normalize(batch[0][0]), batch[0][1]) # account for the domain tag