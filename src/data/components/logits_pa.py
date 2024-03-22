import torch
from src.data.components import Dataset, LogitsDataset
from src.data.components.collate_functions import MultiEnv_collate_fn
from torch.utils.data import DataLoader, DistributedSampler
from src.pa_metric.pairing import PosteriorAgreementDatasetPairing
from typing import Optional

import gc # garbage collector for the dataset

class LogitsPA:
    """
    Mixin class for PA datamodules that passes the logits to the model instead of the images. This helps speed up the process significantly.
    """

    def __init__(self, classifier: torch.nn.Module):
        self.classifier = classifier

    def _logits_dataset(self, image_dataset: Dataset):
        self.classifier.eval()
        self.dev = next(self.classifier.parameters()).device # wherever the model is

        # TODO: Check if this is the best option
        # Batch size is for training, not for evaluating, so it should be higher.
        # Num workers might give us problems when we are on lightning DDP.
        image_dataloader = DataLoader(
            dataset=image_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=MultiEnv_collate_fn,
            shuffle=False, 
            drop_last = False
        )

        with torch.no_grad():
            logits = []
            ys = []
            for bidx, batch in enumerate(image_dataloader):
                envs = list(batch.keys())
                X_list = [batch[envs[e]][0] for e in range(self.num_envs)]
                Y_list = [batch[envs[e]][1] for e in range(self.num_envs)]
                if not all([torch.equal(Y_list[0], Y_list[i]) for i in range(1, len(Y_list))]): # all labels must be equal
                    raise ValueError("The labels of the two environments must be the same.")

                logits.append([self.classifier(X.to(self.dev)) for X in X_list]) 
                ys.append(Y_list[0])
        
        lds = LogitsDataset(
            [torch.cat([logits[i][e] for i in range(len(ys))]) for e in range(self.num_envs)],
            torch.cat(ys)
        )
        return lds
    
    def setup(self, stage: Optional[str] = None):
        super().setup(stage)
        self.logits_ds = self._logits_dataset(self.train_ds)

        # Free up memory asap
        self.train_ds = None
        gc.collect()

    def train_dataloader(self):
        return DataLoader(
                dataset=self.logits_ds,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                collate_fn=MultiEnv_collate_fn,
                shuffle=True, 
                drop_last = False
            )