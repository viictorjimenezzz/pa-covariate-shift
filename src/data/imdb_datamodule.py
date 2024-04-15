from typing import Optional
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer

from transformers import DataCollatorWithPadding

class IMDBDataModule(LightningDataModule):
    def __init__(
            self,
            tokenizer,
            n_classes: int,
            n_train: Optional[int] = 20000,
            n_val: Optional[int] = 5000,
            n_test: Optional[int] = 25000,
            batch_size: Optional[int] = 64,
            num_workers: Optional[int] = 0,
            seed: Optional[int] = 42
        ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["tokenizer"])
        self.tokenizer = tokenizer
        self.data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
        self.hparams.n_train = min(n_train, 20000)
        self.hparams.n_val = min(n_val, 5000)
        self.hparams.n_test = min(n_test, 25000)

    @property
    def num_classes(self):
        return self.hparams.n_classes

    def prepare_data(self):
        load_dataset("imdb") # download if required

    def _tokenize(self, item):
        return self.tokenizer(
            item["text"],
            truncation=True,
            # padding="max_length",
            # max_length=512
        )

    def setup(self, stage=None):
        imdb = load_dataset("imdb")
        if stage == "fit":
            trainval_ds = imdb["train"].shuffle(seed=self.hparams.seed)

            self.train_ds = trainval_ds.select(
                [i for i in list(range(self.hparams.n_train))]
            ).map(self._tokenize, batched=True)

            import ipdb; ipdb.set_trace()

            self.val_ds = trainval_ds.select(
                [i for i in list(range(self.hparams.n_train, self.hparams.n_train + self.hparams.n_val))]
            ).map(self._tokenize, batched=True)

        elif stage == "test":
            self.test_ds = imdb["test"].shuffle(seed=self.hparams.seed).select(
                [i for i in list(range(self.hparams.n_test))]
            ).map(self._tokenize, batched=True)

    def _collate_fn(self, batch):
        return self.data_collator(batch)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self._collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self._collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self._collate_fn
        )