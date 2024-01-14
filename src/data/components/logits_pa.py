import torch
from src.data.components import LogitsDataset
from src.data.components.collate_functions import MultiEnv_collate_fn
from torch.utils.data import DataLoader, DistributedSampler
from src.pa_metric_torch import PosteriorAgreementSampler

class LogitsPA:
    """
    Mixin class for PA datamodules that passes the logits to the model instead of the images. This helps speed up the process significantly.
    """

    def __init__(self, classifier: torch.nn.Module):
        self.classifier = classifier
        self.dev = "cuda" if torch.cuda.is_available() else "cpu" # modify to set device

    def _logits_dataset(self, image_dataloader: DataLoader):
        self.classifier.eval()
        with torch.no_grad():
            logits = []
            ys = []
            for bidx, batch in enumerate(image_dataloader):
                envs = batch["envs"]
                X_list = [batch[envs[e]][0] for e in range(self.num_envs)]
                Y_list = [batch[envs[e]][1] for e in range(self.num_envs)]
                if not all([torch.equal(Y_list[0], Y_list[i]) for i in range(1, len(Y_list))]): # all labels must be equal
                    raise ValueError("The labels of the two environments must be the same.")

                logits.append([self.classifier(X.to(self.dev)) for X in X_list]) 
                ys.append(Y_list[0])
        
        return LogitsDataset(
            [torch.cat([logits[bidx][e] for bidx in range(len(ys))]) for e in range(self.num_envs)],
            torch.cat(ys)
        )
    
    def _set_sampler(self):
        """For the super().train_dataloader()"""
        # Because I operate within the GPU, but still want pairing
        return PosteriorAgreementSampler(self.test_pairedds, shuffle=False, drop_last = True, num_replicas=1, rank=0)

    def train_dataloader(self):
        logits_dataset = self._logits_dataset(super().train_dataloader()) # unconventional here but still GPU
        return DataLoader(
                dataset=logits_dataset,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                collate_fn=MultiEnv_collate_fn,
                sampler=DistributedSampler(logits_dataset, shuffle=False, drop_last = False) # samples already paired with PosteriorAgreementSampler
            )

    # val_dataloader should be the same as train_dataloader in the main class