from typing import Optional, List
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, LightningModule
from pametric.lightning.callbacks import MeasureOutput_Callback

class PCA_Callback(MeasureOutput_Callback):
    """
    Obtain a plot E1vsE2 in the PCA direction. The PCA is done separatedly for every environment, since then we can argue that the variation
    between samples has nothing to do with the distribution shift. In other words, the latent variable that the PCA direction represents
    is (or should be) the label and nothing else.
    """
    def __init__(self, *args, on_epochs: Optional[List[int]] = [0], plot: Optional[bool] = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_features = True
        self.average = False
        self.on_epochs = on_epochs
        self.plot = plot

    def _iterate_and_sum(self, dataloader: DataLoader, model_to_eval: nn.Module) -> torch.Tensor:
        dataloader = self._reinstantiate_dataloader(dataloader)

        output, labels = [], []
        for bidx, batch in enumerate(dataloader):
            if bidx == 0:
                self.list_envs = list(batch.keys())
        
            # Here depends wether the features have to be extracted or not
            output.append([
                model_to_eval.forward(batch[e][0], self.output_features)
                for e in self.list_envs
            ])
            # We assume sample correspondence:
            labels.append(batch[self.list_envs[0]][1])

        output = [
            torch.cat([
                out[e] for out in output
            ])
            for e in range(self.num_envs)
        ]  
        labels = torch.cat(labels)
          
        X_e = []
        for e in range(min(self.num_envs, 2)):
            features_normalized = (output[e] - output[e].mean(dim=0)) / output[e].std(dim=0)
            # We perform PCA for samples in the same environment
            X_e.append(
                self._normalize_pca(
                    self._pca(features_normalized.nan_to_num(), 1)
                )
            )

        del output
        return self._pca_metrics(self._invert_coordinates(X_e), labels)

    def _pca(self, X, num_components: int = 2):
        covariance_matrix = torch.mm(X.t(), X) / (X.size(0) - 1)

        eigenvalues, eigenvectors = torch.eig(covariance_matrix, eigenvectors=True)
        eigenvalues = eigenvalues[:, 0]  # Extract real part of eigenvalues

        sorted_indices = torch.argsort(eigenvalues, descending=True)
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        principal_components = sorted_eigenvectors[:, :num_components]
        return torch.mm(X, principal_components)
    
    def _normalize_pca(self, X_pca):
        X_min = X_pca.min(dim=0, keepdim=True)[0]
        X_max = X_pca.max(dim=0, keepdim=True)[0]
        return 2 * (X_pca - X_min) / (X_max - X_min) - 1

    def _covmatrix(self, X_2):
        X_2_centered = X_2 - X_2.mean(dim=0)
        covariance_matrix = torch.mm(X_2_centered.t(), X_2_centered)/(X_2.size(0) - 1)
        return covariance_matrix
    
    def _invert_coordinates(self, X_2d):
        """
        Invert coordinate so that big cluster is always in the lower left corner (convention).
        """
        if (X_2d[0] < 0.0).sum() < len(X_2d[0])/2:
            X_2d[0] *= -1
        if (X_2d[1] < 0.0).sum() < len(X_2d[1])/2:
            X_2d[1] *= -1
        return X_2d

    def _pca_metrics(self, X_e: torch.Tensor, labels: torch.Tensor):
        covs_lab = []
        for e in range(self.num_envs):
            for lab in torch.unique(labels):
                mask = labels == lab.item()
                covs_lab.append(
                    self._covmatrix(torch.cat([X_e[0][mask], X_e[1][mask]], dim=1))
                )

        # Compote the mean square error from the identity line: we must normalize first
        mse = (X_e[1] - X_e[0]).pow(2).mean()
        cov = self._covmatrix(torch.cat(X_e, dim=1))
        return mse, cov, covs_lab

    def _log_average(self, logger, pca_metrics: tuple, current_epoch: int):
        mse, cov, covs_lab = pca_metrics

        dict_to_log = {
                f"PA/MSE": mse.item()
        }
        self.log_dict(dict_to_log, prog_bar=False, on_step=False, on_epoch=True, logger=True, sync_dist=False)

        columns = ["0", "1"]
        logger.log_table(key=f"COV_{current_epoch}", columns=columns, data=cov.tolist())
        for ilab, cov in enumerate(covs_lab):
            logger.log_table(key=f"COV@{ilab}_{current_epoch}", columns=columns, data=cov.tolist())

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        if pl_module.current_epoch in self.on_epochs:
            pca_metrics = self._compute_average(trainer=trainer, pl_module=pl_module)
            self._log_average(pl_module.logger, pca_metrics, pl_module.current_epoch)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        pca_metrics = self._compute_average(trainer=trainer, pl_module=pl_module)
        self._log_average(pca_metrics, pl_module.current_epoch)
    




# from tsne_torch import TorchTSNE as TSNE
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
# https://github.com/palle-k/tsne-pytorch

class BidimensionalEmbeddingsPlot_Callback(MeasureOutput_Callback):
    def __init__(self, on_epochs: Optional[List[int]] = [0], tsne: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_features = True
        self.average = False # Important to avoid error.
        self.on_epochs = on_epochs
        self.tsne = tsne

    def _iterate_and_sum(self, dataloader: DataLoader, model_to_eval: nn.Module) -> torch.Tensor:
        dataloader = self._reinstantiate_dataloader(dataloader)

        output, labels = [], []
        for bidx, batch in enumerate(dataloader):
            if bidx == 0:
                self.list_envs = list(batch.keys())
            # Here depends wether the features have to be extracted or not
            output.append([
                model_to_eval.forward(batch[e][0], self.output_features)
                for e in self.list_envs
            ])
            # We assume sample correspondence:
            labels.append(batch[self.list_envs[0]][1])

        output = [
            torch.cat([
                out[e] for out in output
            ])
            for e in range(self.num_envs)
        ]  
        labels = torch.cat(labels)
          
        X_e = []
        for e in range(self.num_envs):
            features_normalized = (output[e] - output[e].mean(dim=0)) / output[e].std(dim=0)

            # We perform PCA for samples in the same environment
            X_e.append(
                self._normalize_pca(
                    self._pca(features_normalized.nan_to_num(), 1)
                )
            )
            
            
            if e == 1 and self.tsne:
                output_tsne = torch.cat(output[:2], dim=0)
                features_normalized = (output_tsne - output_tsne.mean(dim=0)) / output_tsne.std(dim=0)

                # # DEBUGGING ------------------------------------------------------------------------------
                # features_normalized_debug = torch.cat([features_normalized]*3, dim=0)
                # X_tsne = TSNE(n_components=2, perplexity=20, n_iter=100, verbose=True).fit_transform(features_normalized_debug)
                # # ----------------------------------------------------------------------------------------

                # Sample to avoid CUDA OOM error:
                # from src.callbacks.utils import rows_that_fit_to_cuda_memory
                # rows_to_keep = rows_that_fit_to_cuda_memory(features_normalized)
                # tsne_indices = np.random.choice(output[0].size(0), int(output[0].size(0)*0.4), replace=False)
                # tsne_indices_double = np.concatenate((tsne_indices, tsne_indices+output[0].size(0))) # pairwise
                # features_normalized = features_normalized[tsne_indices_double] # renamed

                X_tsne = TSNE(n_components=2, perplexity=20, n_iter=100, verbose=False).fit_transform(features_normalized)
                # Separate by environments for the plot
                X_tsne = [
                    X_tsne[:output[0].size(0), :], X_tsne[output[0].size(0):, :]
                ]
                fig_tsne = self._plot_tsne(X_tsne, labels)
                # ----------------------------------------------------------------------------------------
                # X_tsne = TSNE(n_components=2, perplexity=64, n_iter=1000, verbose=False).fit_transform(features_normalized)

        
        fig_pca, metrics_pca = self._plot_pca(self._invert_coordinates(X_e), labels)
        fig_tsne = None
        if self.tsne:
            fig_tsne = self._plot_tsne(self._invert_coordinates(X_tsne), labels)
        return (fig_pca, fig_tsne), metrics_pca

    def _pca(self, X, num_components: int = 2):
        covariance_matrix = torch.mm(X.t(), X) / (X.size(0) - 1)

        eigenvalues, eigenvectors = torch.eig(covariance_matrix, eigenvectors=True)
        eigenvalues = eigenvalues[:, 0]  # Extract real part of eigenvalues

        sorted_indices = torch.argsort(eigenvalues, descending=True)
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        principal_components = sorted_eigenvectors[:, :num_components]
        return torch.mm(X, principal_components)

    def _normalize_pca(self, X_pca):
        X_min = X_pca.min(dim=0, keepdim=True)[0]
        X_max = X_pca.max(dim=0, keepdim=True)[0]
        return 2 * (X_pca - X_min) / (X_max - X_min) - 1

    def _invert_coordinates(self, X_2d):
        """
        Invert coordinate so that big cluster is always in the lower left corner (convention).
        """
        if (X_2d[0] < 0.0).sum() < len(X_2d[0])/2:
            X_2d[0] *= -1
        if (X_2d[1] < 0.0).sum() < len(X_2d[1])/2:
            X_2d[1] *= -1
        return X_2d
    
    def _plot_pca(self, X_e: torch.Tensor, labels: torch.Tensor):
        # Get the font
        fontname = "DejaVu Serif"
        _ = fm.findfont(fm.FontProperties(family=fontname))

        # markers = ['o', '^', 's', 'p', '*', '+', 'x', 'D', 'h', '|']
        colors_list = ['tab:red', 'tab:blue', 'tab:green', 'tab:purple', 'tab:orange', 'tab:cyan']
        
        data, covs_lab = [], []
        for e in range(self.num_envs):
            for ilab, lab in enumerate(torch.unique(labels)):
                mask = labels == lab.item()
                covs_lab.append(
                    self._covmatrix(torch.cat([X_e[0][mask], X_e[1][mask]], dim=1))
                )
                # ax.scatter(
                #     X_e[0][mask].cpu(),
                #     X_e[1][mask].cpu(),
                #     color=colors_list[ilab],
                #     label=f"Label {lab.item()}" if e == 0 else None
                # )

                x_data = X_e[0][mask].cpu().numpy().squeeze()
                y_data = X_e[1][mask].cpu().numpy().squeeze()
                label_data = np.array([f"Label {lab.item()}"] * len(x_data))
                data.append(pd.DataFrame({
                    'PCA_Env_0': x_data,
                    'PCA_Env_1': y_data,
                    'Label': label_data
                }))

        df = pd.concat(data, ignore_index=True)

        for iterate_plot in range(2): # for some reason it doesnt work at the first.
            fig, ax = plt.subplots(figsize=(2 * 3.861, 2 * 2.7291))
            sns.set(font_scale=1.9)
            plt.rcParams["font.family"] = "serif"
            plt.rcParams["font.serif"] = fontname
            sns.set_style("ticks")

            sns.scatterplot(
                data=df,
                x='PCA_Env_0',
                y='PCA_Env_1',
                hue='Label',
                palette=colors_list,
                ax=ax,
                s=100,  # Adjust marker size
                edgecolor="w"
            )

            # Compote the mean square error from the identity line: we must normalize first
            mse = (X_e[1] - X_e[0]).pow(2).mean()
            cov = self._covmatrix(torch.cat(X_e, dim=1))
            ax.plot(
                [X_e[0].min().item(), X_e[0].max().item()], [X_e[0].min().item(), X_e[0].max().item()],
                color='k',
                ls = "--"
            )
            ax.set_title(f"PCA Epoch {self.current_epoch}", fontname=fontname)
            ax.set_xlabel(f'PC Environment 0', fontname=fontname)
            ax.set_ylabel(f'PC Environment 1', fontname=fontname)
            ax.legend(
                loc= "best",
                prop = {
                    "family": fontname,
                    'size': 18
                }
            )
            ax.grid(True)
            ax.grid(linestyle="--")

            ax.minorticks_on()
            ax.tick_params(axis="both", which="both", direction="in")
            xticks_font = fm.FontProperties(family=fontname)
            for tick in ax.get_xticklabels():
                tick.set_fontproperties(xticks_font)

            # Save the figure
            fig.savefig(f"pca_plot_{self.current_epoch}.png", dpi=300, bbox_inches='tight', pad_inches=0.1)

        print(f'\n\n100MSE: {100.0*mse:.3f}, 100DetCOV: {100.0*cov.det():.2f}')
        with open(r"/cluster/home/vjimenez/adv_pa_new/pca_plot_logs.txt", 'a') as file:
            file.write(f'\nEPOCH: {self.current_epoch} - 100MSE: {100.0*mse}, 100DetCOV: {100.0*cov.det()}')

        return fig, (mse, cov, covs_lab)
    

    def _plot_tsne(self, X_tsne: np.ndarray, labels: np.ndarray):
        markers_list = ['o', '^', 's', 'p', '*', '+', 'x', 'D', 'h', '|']
        colors_list = ['tab:red', 'tab:blue', 'tab:green', 'tab:purple', 'tab:orange', 'tab:cyan']

        fig, ax = plt.subplots(figsize=(5, 4))
        covs_lab = []
        for e in range(self.num_envs):
            for ilab, lab in enumerate(torch.unique(labels)):
                mask = labels == lab.item()
                ax.scatter(
                    X_tsne[e][mask,0],
                    X_tsne[e][mask,1],
                    marker=markers_list[ilab],
                    color=colors_list[e],
                    label=f"Environment {e}, Label {lab.item()}"
                )

        # Improve plot aesthetics
        ax.set_title(f'2D t-SNE plot')
        ax.set_xlabel(f't-SNE dimension 0')
        ax.set_ylabel(f't-SNE dimension 1')
        ax.legend(loc="best")
        ax.grid(True)
        fig.savefig(f"tsne_plot.png")
        return fig
    
    def _covmatrix(self, X_2):
        X_2_centered = X_2 - X_2.mean(dim=0)
        covariance_matrix = torch.mm(X_2_centered.t(), X_2_centered)/(X_2.size(0) - 1)
        return covariance_matrix

    def _log_average(self, fig_metrics: tuple, current_epoch: int):
        pass
        # (fig_pca, fig_tsne), (mse, cov, covs_lab) = fig_metrics
        # fig_pca.savefig(f"./pca.png")

        # if self.tsne:
        #     fig_tsne.savefig(f"./tsne.png")


        # dict_to_log = {
        #     f"PA/MSE": mse
        # }
        # self.log_dict(dict_to_log, prog_bar=False, on_step=False, on_epoch=True, logger=True, sync_dist=True)

        # wandb.log({f"COV_{current_epoch}": wandb.Table(dataframe=pd.DataFrame(cov.tolist()))})
        # for ilab, cov in enumerate(covs_lab):
        #     wandb.log({f"COV@{ilab}_{current_epoch}": wandb.Table(dataframe=pd.DataFrame(cov.tolist()))})

        # # Assuming logger is Weights & Biases. Change this otherwise:
        # try:
        #     wandb.log({f"PA/PCA_{current_epoch}": wandb.Image(f"./pca.png")})
        #     if self.tsne:
        #         wandb.log({f"PA/t-SNE_{current_epoch}": wandb.Image(f"./tsne.png")})
        # except:
        #     print("Image could not be logged. Please specify the logging format.")

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        if pl_module.current_epoch in self.on_epochs:
            self.current_epoch = pl_module.current_epoch
            figs = self._compute_average(trainer=trainer, pl_module=pl_module)
            self._log_average(figs, pl_module.current_epoch)
    
    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        figs = self._compute_average(trainer=trainer, pl_module=pl_module)
        self._log_average(figs, pl_module.current_epoch)
