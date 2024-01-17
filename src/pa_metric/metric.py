import os
import warnings

import torch
from torchmetrics import Metric
from typing import Optional, List, Union
from torch.utils.data import DataLoader

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import SequentialSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group
from src.data.components import MultienvDataset, LogitsDataset
from src.data.components.collate_functions import MultiEnv_collate_fn

from .sampler import PosteriorAgreementSampler
from .kernel import PosteriorAgreementKernel

#TODO: Check this out
    # # Set to True if the metric is differentiable else set to False
    # is_differentiable: Optional[bool] = None

    # # Set to True if the metric reaches it optimal value when the metric is maximized.
    # # Set to False if it when the metric is minimized.
    # higher_is_better: Optional[bool] = True

    # # Set to True if the metric during 'update' requires access to the global metric
    # # state for its calculations. If not, setting this to False indicates that all
    # # batch states are independent and we will optimize the runtime of 'forward'
    # full_state_update: bool = True

class PosteriorAgreementSimple(Metric):
    def __init__(self, 
                 pa_epochs: int,
                 beta0: Optional[float] = None):
        super().__init__()

        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.beta0 = beta0
        self.pa_epochs = pa_epochs

        # Preallocate metrics to track
        self.afr_true = torch.zeros(self.pa_epochs).to(self.dev) # metrics live in master process device
        self.afr_pred = torch.zeros_like(self.afr_true)
        self.accuracy = torch.zeros_like(self.afr_true)

        # Kernel and optimizer are initialized right away
        self.kernel = PosteriorAgreementKernel(beta0=self.beta0).to(self.dev)
        self.optimizer = torch.optim.Adam([self.kernel.module.beta], lr=0.01)

    def update(self, logits_dataloader: DataLoader):
        """
        For this simple version, the logits will be passed and the kernel optimized.
        """
        
        self.betas = torch.zeros_like(self.afr_true)
        self.logPAs = torch.full_like(self.afr_true, -float('inf'))
        for epoch in range(self.pa_epochs):
            for bidx, batch in enumerate(logits_dataloader):
                self.kernel.module.beta.data.clamp_(min=0.0)
                self.kernel.module.reset()

                envs = batch["envs"]
                logits0, logits1 = batch[envs[0]][0], batch[envs[1]][0]
                with torch.set_grad_enabled(True):
                    loss = self.kernel.module.forward(logits0, logits1)  
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            self.kernel.module.beta.data.clamp_(min=0.0) # project to >=0 one last time
            beta_last = self.kernel.module.beta.item()
            self.betas[epoch] = beta_last

            # Compute logPA with the last beta per epoch
            self.kernel.module.reset()
            correct, correct_pred, correct_true = 0, 0, 0
            for bidx, batch in enumerate(logits_dataloader):
                envs = batch["envs"]
                logits0, logits1, y = batch[envs[0]][0], batch[envs[1]][0], batch[envs[0]][1]

                # Compute accuracy metrics if desired
                if y and epoch == 0:
                    y_pred = torch.argmax(logits0.to(self.dev), 1) # env 1
                    y_pred_adv = torch.argmax(logits1.to(self.dev), 1) # env 2
                    correct_pred += (y_pred_adv == y_pred).sum().item()
                    correct_true += (y_pred_adv == y).sum().item()
                    correct += (torch.cat([y_pred, y_pred_adv]).to(self.dev) == torch.cat([y, y]).to(self.dev)).sum().item()

                # Update logPA
                self.kernel.module.evaluate(beta_last, logits0, logits1)

            # Retrieve final logPA
            self.logPAs[epoch] = self.kernel.module.log_posterior().item()

            if y and epoch == 0: # retrieve accuracy metrics
                self.afr_pred[epoch] = correct_pred/len(logits0)
                self.afr_true[epoch] = correct_true/len(logits0)
                self.accuracy[epoch] = correct/(2*len(logits0))

        # Locate the highest PA.
        self.selected_index = torch.argmax(self.logPAs).item()

    def compute(self):
        """
        Only meant to be used at the end of the PA optimization.
        """
        return {
            "beta": self.betas[self.selected_index],
            "logPA": self.logPAs[self.selected_index],
            "PA": torch.exp(self.logPAs[self.selected_index]), # TODO: Fix this small error
            "AFR pred": self.afr_pred[self.selected_index],
            "AFR true": self.afr_true[self.selected_index],
            "acc_pa": self.accuracy[self.selected_index]
        }


class PosteriorAgreement(PosteriorAgreementSimple):
    def __init__(self, 
                 dataset: MultienvDataset,
                 early_stopping: Optional[List] = None,
                 strategy: Optional[str] = "cuda",
                 cuda_devices: Optional[Union[List[str], int]] = None,
                 *args, **kwargs):
        
        if strategy not in ["cuda", "cpu", "lightning"]:
            raise ValueError("The strategy must be either 'cuda', 'cpu' or 'lightning'.")
        
        super().__init__(*args, **kwargs)

        self.strategy = strategy
        
        # Initialize multiprocessing configuration
        self.ddp_init = None
        if self.strategy != "lightning": # cuda or cpu
            if dist.is_initialized(): # ongoing cuda
                self.device_list = [f"cuda:{i}" for i in range(dist.get_world_size())]
            else: # non initialized cuda or cpu
                if cuda_devices:
                    if isinstance(cuda_devices, int):
                        cuda_devices = [f"cuda:{i}" for i in range(cuda_devices)]
                    self.device_list = cuda_devices if (torch.cuda.is_available() and self.strategy == "cuda") else ["cpu"]
                else:
                    self.device_list = ["cuda"] if (torch.cuda.is_available() and self.strategy == "cuda") else ["cpu"]
                self.ddp_init = [False]*len(self.device_list) if "cuda" in self.device_list[0] else None
            self.dev = self.device_list[0]
        else: # Depending where the metric is initialized we will have to update it or not, but the accuracy tensors are already here
            self.dev = "cuda" if dist.is_initialized() else "cpu"
            self.device_list = [self.dev]

        print("IS CUDA AVAILABLE: ", torch.cuda.is_available())
        print("device list: ", self.device_list)
        print("dev: ", self.dev)
        print("ddp_init: ", self.ddp_init)

        # Check dataloader conditions
        if not isinstance(dataset, MultienvDataset):
            raise ValueError("The dataloader must be wrapped using a MultienvDataset.")
        
        self.dataloader = DataLoader(
            dataset=dataset,
            sampler=PosteriorAgreementSampler(dataset, shuffle=False, drop_last=True, num_replicas=len(self.device_list), rank=0),
            collate_fn=MultiEnv_collate_fn,
            shuffle=False, # we use custom sampler

            # Decide whether this has to be set as config input or not
            batch_size = 64,
            num_workers = 0, # 4*len(self.device_list) if "cuda" in self.dev else max(2, min(8, os.cpu_count())),
            pin_memory = ("cuda" in self.dev),
        )    
        self.num_envs = self.dataloader.sampler.dataset.num_envs # get the modified version
        self.batch_size = self.dataloader.batch_size  
        self.num_batches = self.dataloader.sampler.num_samples // self.batch_size

        # Define early stopping parameters
        self.tol, self.itertol, self.patience = None, None, 0
        if early_stopping:
            self.tol = early_stopping[0]*torch.ones(early_stopping[1]).to(self.dev) # tensor([tol, tol, tol, ...])
            self.itertol = float('inf')*torch.ones(early_stopping[1]).to(self.dev) # tensor([inf, inf, inf, ...]) to be filled with relative variations of beta
            self.patience = early_stopping[2]

            # Checking inputs
            if self.patience > self.pa_epochs:
                warnings.warn("The patience is greater than the number of epochs. Early stopping will not be applied.")
                self.tol, self.itertol, self.patience = None, None, 0
            if early_stopping[1] > self.pa_epochs:
                warnings.warn("The number of iterations to consider for early stopping is greater than the number of epochs. Early stopping will not be applied.")
                self.tol, self.itertol, self.patience = None, None, 0

    def _logits_dataset(self, dev, classifier: torch.nn.Module, classifier2: Optional[torch.nn.Module] = None):
        classifier.to(dev)
        if classifier2:
            classifier2.to(dev)

        y_totensor = [None]*len(self.dataloader)
        X_totensor = [None]*len(self.dataloader)
        for bidx, batch in enumerate(self.dataloader):
            if bidx == 0: # initialize logits dataset
                envs = batch["envs"]
                if len(envs) != self.num_envs:
                    raise ValueError("There is a problem with the configuration of the Dataset and/or the DataLoader collate function.")
                
            X_list = [batch[envs[e]][0].to(dev) for e in range(self.num_envs)]
            Y_list = [batch[envs[e]][1].to(dev) for e in range(self.num_envs)]
            if not all([torch.equal(Y_list[0], Y_list[i]) for i in range(1, len(Y_list))]): # all labels must be equal
                raise ValueError("The labels of the two environments must be the same.")
            
            y_totensor[bidx] = Y_list[0]
            if classifier2: # then the validation with additional datasets uses the second classifier
                X_totensor[bidx] = [classifier(X_list[0])] + [classifier2(X_list[i]) for i in range(1, len(X_list))]
            else: # subset has two elements, each with the same labels
                X_totensor[bidx] = [classifier(X) for X in X_list]

        logits_list = [torch.cat([X_totensor[j][i] for j in range(len(self.dataloader))]) for i in range(len(X_list))]
        y = torch.cat(y_totensor)

        return LogitsDataset(logits_list, y)    
    
    def _pa_validation(self, dev, kernel, fixed_beta, env, logits_dataloader: DataLoader):
        kernel.module.reset()
        total_samples = 0
        correct, correct_pred, correct_true = 0, 0, 0
        with torch.no_grad():
            for bidx, batch in enumerate(logits_dataloader):
                # This is the return of the __getitem__ method. No need of a collate function bc this will not generalize.
                #{str(i): tuple([self.logits[i][index], self.y[index]]) for i in range(self.num_envs)}

                logit0, y = batch['0'][0], batch['0'][1]
                logit1 = batch[str(env)][0]
                    
                # Compute accuracy metrics
                y_pred = torch.argmax(logit0.to(dev), 1) # env 1
                y_pred_adv = torch.argmax(logit1.to(dev), 1) # env 2
                correct_pred += (y_pred_adv == y_pred).sum().item()
                correct_true += (y_pred_adv == y).sum().item()
                correct += (torch.cat([y_pred, y_pred_adv]).to(dev) == torch.cat([y, y]).to(dev)).sum().item()
                total_samples += len(y)

                # Update logPA
                kernel.module.evaluate(fixed_beta, logit0, logit1)
            
            # Retrieve final logPA for the (subset) batches
            logPA = kernel.module.log_posterior().to(dev) 

            # Retrieve logPA and accuracy metrics for the epoch and log
            if "cuda" in dev and self.strategy == "cuda":
                dist.all_reduce(logPA, op=dist.ReduceOp.SUM)
                dist.all_reduce(torch.tensor(total_samples).to(dev), op=dist.ReduceOp.SUM)
                dist.all_reduce(torch.tensor(correct_pred).to(dev), op=dist.ReduceOp.SUM)
                dist.all_reduce(torch.tensor(correct_true).to(dev), op=dist.ReduceOp.SUM)
                dist.all_reduce(torch.tensor(correct).to(dev), op=dist.ReduceOp.SUM)

            return {
                "logPA": logPA.item(),
                "AFR pred": correct_pred/total_samples,
                "AFR true": correct_true/total_samples,
                "accuracy": correct/(2*total_samples)
            }

    def _optimize_beta(self, rank: int, classifier: torch.nn.Module, classifier2: Optional[torch.nn.Module] = None):
        if self.strategy == "lightning":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        else: 
            dev = str(self.device_list[rank])  
            
        self.dataloader.sampler.rank = rank # adjust to device to be used
        logits_dataset = self._logits_dataset(dev, classifier, classifier2)
        logits_dataloader = DataLoader(dataset=logits_dataset,
                                       batch_size=self.batch_size, # same as the data
                                       num_workers=0, # we won't create subprocesses inside a subprocess, and data is very light
                                       pin_memory=False, # only dense CPU tensors can be pinned

                                       # Important so that it matches with the input data.
                                       shuffle=False,
                                       drop_last = False,
                                       sampler=SequentialSampler(logits_dataset))

        # load training objects every time
        kernel = PosteriorAgreementKernel(beta0=self.beta0).to(dev)
        if "cuda" in dev and self.strategy == "cuda":
            kernel = DDP(kernel, device_ids=[dev])
        optimizer = torch.optim.Adam([kernel.module.beta], lr=0.01)

        # Optimize beta for every batch within an epoch, for every epoch
        for epoch in range(self.pa_epochs):
            beta_e = 0.0
            for bidx, batch in enumerate(logits_dataloader):
                kernel.module.beta.data.clamp_(min=0.0)
                kernel.module.reset()

                logits, _ = batch
                with torch.set_grad_enabled(True):
                    loss = kernel.module.forward(logits[0].to(dev), logits[1].to(dev))  
                    beta_e += kernel.module.beta.item()

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

            # Retrieve betas and compute the mean over the epoch
            if "cuda" in dev and self.strategy == "cuda":
                dist.all_reduce(torch.tensor(beta_e).to(dev), op=dist.ReduceOp.SUM) # sum betas from all processes for the same epoch
            beta_mean = beta_e / self.num_batches
            self.betas[epoch] = beta_mean

            # Compute logPA with the mean beta for the epoch and validate
            for i in range(1, self.num_envs):
                metric_dict = self._pa_validation(dev, kernel, beta_mean, i, logits_dataloader)
                if i == 1: # the ones for the first environment must be stored
                    self.logPAs[epoch] = metric_dict["logPA"]
                    self.afr_pred[epoch] = metric_dict["AFR pred"]
                    self.afr_true[epoch] = metric_dict["AFR true"]
                    self.accuracy[epoch] = metric_dict["accuracy"]

                else:
                    if epoch == self.pa_epochs-1: # TODO: Decide what to do with this
                        print(metric_dict)
                        with open('logs_pa_metric.txt', 'a') as log_file:
                            log_file.writelines([f"metric dict for 0-{i}" + str(metric_dict) + "\n"])

            # Check for beta relative variation and implement early stopping
            if self.tol != None and epoch > self.patience: 
                relvar = torch.tensor([abs(beta_mean - self.betas[epoch-1])/beta_mean]).to(self.dev)
                self.itertol = torch.cat([self.itertol[1:], relvar]).to(self.dev)
                if torch.le(self.itertol, self.tol).all().item():
                    print(f"PA optimization stopped at epoch {epoch}.")
                    break

    def _init_DDP_wrapper(self, rank: int, classifier: torch.nn.Module, classifier2: Optional[torch.nn.Module] = None):
        """
        Implements optimization after initializing the corresponding subprocesses.
        """
        if self.ddp_init[rank] == False: # Initialize the process only once, even if the .update() is called several times during a training procedure.
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ["MASTER_PORT"] = "50000"
            init_process_group(backend="nccl", rank=rank, world_size=len(self.device_list))
            torch.cuda.set_device(rank)
            self.ddp_init[rank] = True

        self._optimize_beta(rank, classifier, classifier2)

    def update(self, classifier: torch.nn.Module, classifier2: Optional[torch.nn.Module] = None, destroy_process_group: Optional[bool] = False):
        """
        The goal is to make the Metric as versatile as possible. The Metric can be called in two ways:
        - During a training procedure. In such case, it will use the training strategy already in place (e.g DDP).
        - With a trained model, for evaluation. In such case, the training strategy can be selected: CPU or (multi)-GPU with DDP.

        Important: If used during training, pass a copy.deepcopy() of the classifier(s) to avoid errors.
        """

        # Set to eval mode and freeze parameters
        classifier.eval()
        for param in classifier.parameters():
            param.requires_grad = False
        if classifier2:
            classifier2.eval()
            for param in classifier2.parameters():
                param.requires_grad = False

        # Optimize beta depending on the strategy and the devices available
        if dist.is_initialized(): # ongoing cuda or ddp lightning
            self._optimize_beta(dist.get_rank(), classifier, classifier2)
        else:
            if "cuda" in self.dev: # cuda for the metric
                mp.spawn(self._init_DDP_wrapper,
                    args=(classifier, classifier2,),
                    nprocs=len(self.device_list),
                    join=True) # this gave error
                
                # Set to True when it's the last call to .update()
                if destroy_process_group and dist.is_initialized():
                    dist.destroy_process_group()
            else: # "cpu", either lightning or not
                self._optimize_beta(0, classifier, classifier2)

        # Get the epoch achieving maximum logPA
        self.selected_index = torch.argmax(self.logPAs).item()