from pytorch_lightning.callbacks import Callback
from pytorch_lightning import Trainer, LightningModule

import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from pametric.datautils import MultiEnv_collate_fn

import numpy as np
from copy import deepcopy
from pametric.lightning import SplitClassifier

from faiss import IndexFlatL2, IndexIVFFlat, METRIC_L2

import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler

class FeaturePairing_Callback(Callback):
    """
    Implements epoch-wise pairing of data.
    """

    def _extract_features(
            self,
            dataset: Dataset,
            feature_extractor: torch.nn.Module
        ):
        len_ds = len(dataset)
        features = torch.zeros(( 
            self.num_envs,
            len_ds,
            feature_extractor.forward(dataset[0]['0'][0].unsqueeze(0), extract_features = True).size(1)
        ))

        dataloader = DataLoader(
                    dataset = dataset,
                    collate_fn = MultiEnv_collate_fn,
                    batch_size = self.batch_size,
                    num_workers = 0, 
                    pin_memory = False,
                    sampler = SequentialSampler(dataset),
                    drop_last=False,
        )
        with torch.no_grad():
            for bidx, batch in enumerate(dataloader):
                features[:, bidx*self.batch_size: min((bidx+1)*self.batch_size, len_ds), :] = torch.stack(
                    [
                    feature_extractor.forward(batch[e][0], extract_features = True).squeeze()
                    for e in list(batch.keys())
                ],
                )
            return features

    def _pair_start(self, features: torch.Tensor):
        """
        Override with strategy to deal with features before pairing.
        """
        return
    
    def _pair_end(self, pametric_callback: Callback, permutation: torch.Tensor):
        """
        Override with strategy to deal with permutation after pairing.
        """
        # Finally, apply a random permutation:
        random_perm = torch.randperm(permutation[0].size(0))
        pametric_callback.pa_metric.dataset.permutation = [
            perm[permutation[i][random_perm]]
            for i, perm in enumerate(pametric_callback.pa_metric.dataset.permutation)
        ]
        return
    
    def _pair_environment(self, features_ref: torch.Tensor, features_large: torch.Tensor) -> torch.Tensor:
        """
        Override with strategy to pair feature environments.

        Args:
            features_ref (torch.Tensor): Features of the reference environment.
            features_large (torch.Tensor): Features of the (possibly larger) environment.
        """
        return torch.tensor(np.arange(features_ref.shape[0]), dtype=torch.long)

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # Get the model and split it into feature extractor and classifier
        model_to_eval = SplitClassifier(
            net = deepcopy(pl_module.model.net),
            net_name = pl_module.model.net_name
        ).eval()

        # Get the dataset used by the PA metric, that has already been instantiated (i.e. paired)
        callback_names = [cb.__class__.__name__ for cb in trainer.callbacks]
        pametric_callback = trainer.callbacks[callback_names.index("PA_Callback")]
        dataset = pametric_callback.pa_metric.dataset
        self.num_envs = dataset.num_envs
        self.batch_size = pametric_callback.pa_metric.batch_size

        # Extract the tensor of features: [n_envs, n_samples, dim_features]
        features = self._extract_features(dataset, model_to_eval)

        self._pair_start(features)

        # Perform the pairing: label and NN
        permutation = [torch.arange(len(dataset)).to(dtype=torch.long)]
        labels = dataset.__getlabels__(range(len(dataset)))
        for e in range(1, self.num_envs):
            permutation_e = np.zeros(len(dataset), dtype=int)
            for lab in torch.unique(labels[0]).tolist():
                # Get indices for the current and next element where label matches
                indices_0 = np.where(labels[0] == lab)[0]
                indices_e = np.where(labels[e] == lab)[0]

                # Perform NN pairing on the subset of features matching the label
                indices_local = self._pair_environment(
                    features_ref = features[0, indices_0, :],
                    features_large = features[e, indices_e, :],
                )
                permutation_e[indices_0] = indices_e[indices_local]

            permutation.append(torch.tensor(permutation_e, dtype=torch.long))
        
        self._pair_end(pametric_callback, permutation)


class LabelPairing_Callback(FeaturePairing_Callback):
    """
    It will perform a random iteration between label-corresponding pairs.
    """
    def _pair_environment(self, features_ref: torch.Tensor, features_large: torch.Tensor) -> torch.Tensor:
        return torch.tensor(np.arange(features_ref.shape[0]), dtype=torch.long)


class NNPairing_Callback(FeaturePairing_Callback):

    def __init__(self, index: str = "L2", nearest: bool = True):
        super().__init__()
        self.index = index
        self.nearest = nearest

    def _pair_environment(self, features_ref: torch.Tensor, features_large: torch.Tensor) -> torch.Tensor:
        # See https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index

        features_ref = features_ref.cpu().numpy()
        features_large = features_large.cpu().numpy()
        len_large_ds, len_ref_ds = features_large.shape[0], features_ref.shape[0]
        dim_vecs = features_large.shape[1]

        # TODO: IndexIVFFlat
        # Memory limit of 1 GB for RAM. Each float32 takes 4 bytes.
        RAM_limit = 1 * (1024 ** 3) // (dim_vecs*4) # TODO: adjust

        """
        From the guidelines we deduce that:
        len_large_ds >= n_train >= train_factor*n_clusters = train_factor*cluster_factor*sqrt(len_large_ds)

        The values for the multiplicative factors are:
            - train_factor = 40:256
            - cluster_factor = 4:16

        Then the minimum n_train is 40*4*sqrt(len_large_ds) <= len_large_ds <=> 160 <= sqrt(len_large_ds) <=> 25600 <= len_large_ds
        """

        if self.index == "IVFFlat" and len_large_ds >= 25600:
            """
            Then we train a IVFFlat index.
            """
            # Deciding the number of clusters and index training samples.
            n_train_samples, n_clusters_samples = [1], [2]
            for cluster_factor in range(4, 16):
                for train_factor in range(40, 256): # BUG fix: From 30 to ?? per warning suggestion
                    n_clusters = int(cluster_factor*np.sqrt(len_large_ds)) 
                    n_train = min(
                            RAM_limit, 
                            train_factor*n_clusters, # As per the guidelines
                            len_large_ds # Length of the dataset
                    )
                    if n_train == RAM_limit:
                        break
                    elif n_train >= 40*n_clusters: # safeguard
                        n_train_samples.append(n_train)
                        n_clusters_samples.append(n_clusters)

            pos_max = np.argmax(n_clusters_samples) # the first one will have the fewest number of samples.
            n_train = n_train_samples[pos_max]
            n_clusters = n_clusters_samples[pos_max]
            
            index = IndexIVFFlat(
                IndexFlatL2(dim_vecs),
                dim_vecs,  # Dimension of the vectors
                n_clusters, #int(np.sqrt(large_ds.shape[0])), # Number of clusters
                METRIC_L2 # L2 distance
            )
            train_subset = features_large[np.random.choice(len_large_ds, n_train, replace=False), :]
            index.train(train_subset)

        else:
            """
            Then we use a L2 Flat index.
            """
            index = IndexFlatL2(dim_vecs)

        # Now we can add the data to the index by batches: (10 MB limit)
        batch_size = min(10*(1024 ** 2) // (dim_vecs*4), len_large_ds)
        for i in range(0, len_large_ds, batch_size):
            index.add(
                features_large[i:min(i + batch_size, len_large_ds), :]
            )

        # Quality check:
        _, inds = index.search(features_ref[0, :].reshape(1, -1), k=1) 
        if inds[0].item() == -1:
            raise ValueError("\nThe index has not been trained properly. Try changing the centroid number and the number of training samples.")

        # Perform search for each vector of the reference ds.
        if self.nearest:
            perm1 = torch.tensor(
                np.array([
                index.search(features_ref[i, :].reshape(1, -1), k=1)[1][0].item() # closest element, with repetition
                for i in range(len_ref_ds)
                ])
            )
        else: # The most distant element
            perm1 = torch.tensor(
                np.array([
                index.search(features_ref[i, :].reshape(1, -1), k=len_large_ds)[1][0][-1] # most distant element, with repetition
                for i in range(len_ref_ds)
                ])
            )
        return perm1
    
from torch.utils.data import TensorDataset
from pametric.datautils import MultienvDataset
class CCAPairing_Callback(FeaturePairing_Callback):
    """
    This only works when we only have two environments to match.
    """
    
    def __init__(self):
        super().__init__()
        self.original_dataset = None # It will be stored in the first epoch

    def _pair_environment(self, features_ref: torch.Tensor, features_large: torch.Tensor) -> torch.Tensor:
        scaler_ref = StandardScaler()
        scaler_large = StandardScaler()
        ref_std = scaler_ref.fit_transform(features_ref)
        large_std = scaler_large.fit_transform(features_large)

        # ONLY for debugging purposes ---------------------------------------------------------------------------
        # large_std = np.concatenate([large_std]*4, axis=0)
        # ref_std = np.concatenate([ref_std]*4, axis=0)
        # --------------------------------------------------------------------------------------------------------
        
        cca = CCA(n_components=features_ref.shape[1])
        cca.fit_transform(ref_std, large_std)

        # Project vectors
        ref_projto_large = ref_std @ cca.x_weights_
        large_projto_ref = large_std @ cca.y_weights_
        
        return torch.cat([features_ref, torch.tensor(large_projto_ref)], dim=0), torch.cat([torch.tensor(ref_projto_large), features_large], dim=0)
    
    def _pair_end(self, pametric_callback: Callback, permutation: torch.Tensor):
        return

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """
        We store the original dataset, and we generate a new one to work with PA at every epoch.
        """
        if pl_module.current_epoch == 0:
            callback_names = [cb.__class__.__name__ for cb in trainer.callbacks]
            pametric_callback = trainer.callbacks[callback_names.index("PA_Callback")]
            self.original_dataset = pametric_callback.pa_metric.dataset
            self.num_envs = self.original_dataset.num_envs
            assert self.num_envs == 2, "CCA pairing only works with two environments."
            self.batch_size = pametric_callback.pa_metric.batch_size

        # Get the model and split it into feature extractor and classifier
        model_to_eval = SplitClassifier(
            net = deepcopy(pl_module.model.net),
            net_name = pl_module.model.net_name
        ).eval()

        # Set it as the alternative model for the PA callback.
        pametric_callback.alternative_model = model_to_eval.classifier # only the classifier, which takes the features as an input

        # Extract the tensor of features: [n_envs, n_samples, dim_features]
        features = self._extract_features(self.original_dataset, model_to_eval)

        # Perform the pairing: label and NN
        labels = self.original_dataset.__getlabels__(range(len(self.original_dataset)))
        x_0_ds, x_1_ds, y_0_ds, y_1_ds = [], [], [], []
        for lab in torch.unique(labels[0]).tolist():
            # Get indices for the current and next element where label matches
            indices_0 = np.where(labels[0] == lab)[0]
            indices_e = np.where(labels[1] == lab)[0]

            # Perform NN pairing on the subset of features matching the label
            env_0, env_1 = self._pair_environment(
                features_ref = features[0, indices_0, :],
                features_large = features[1, indices_e, :],
            )
            x_0_ds.append(env_0); x_1_ds.append(env_1)
            y_0_ds.append(lab*torch.ones_like(env_0)); y_1_ds.append(lab*torch.ones_like(env_1))

        # Substitute the dataset: the alternative model will make sure this is processed correctly:
        pametric_callback.pa_metric.dataset = MultienvDataset([
            TensorDataset(
                torch.cat(x_0_ds).to(dtype=torch.float32), 
                torch.cat(y_0_ds).to(dtype=torch.float32)
            ),
            TensorDataset(
                torch.cat(x_1_ds).to(dtype=torch.float32),
                torch.cat(y_1_ds).to(dtype=torch.float32)
            )
        ])
        

# import import import
from typing import Optional

class Pairing_Callback(FeaturePairing_Callback):
    """
    Combine all epoch-wise pairing callbacks into a single one.
    """
    def __init__(self, method: Optional[str] = None, *args, **kwargs):
        self.method = method
        if self.method is not None:
            self.pairing_callback = self._selector(**kwargs)
    
    def _selector(self, **kwargs):
        if self.method.upper() == "NN":
            return NNPairing_Callback(index=kwargs["index"], nearest=kwargs["nearest"])
        elif self.method.upper() == "CCA":
            return CCAPairing_Callback()
        else:
            return LabelPairing_Callback()

    def _pair_start(self, features: torch.Tensor):
        if self.method is not None:
            return self.pairing_callback._pair_start(features)
    
    def _pair_end(self, pametric_callback: Callback, permutation: torch.Tensor):
        if self.method is not None:
            return self.pairing_callback._pair_end(pametric_callback, permutation)
    
    def _pair_environment(self, features_ref: torch.Tensor, features_large: torch.Tensor) -> torch.Tensor:
        if self.method is not None:
            return self.pairing_callback._pair_environment(features_ref, features_large)

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.method is not None:
            return self.pairing_callback.on_train_epoch_start(trainer, pl_module)
        