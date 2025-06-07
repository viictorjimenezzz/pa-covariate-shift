from typing import Optional
import os
import os.path as osp

import torch
from torch.utils.data import TensorDataset

from secml.data.loader import CDataLoaderCIFAR10
from autoattack import AutoAttack
from src.data.utils import carray2tensor


class AdversarialCIFAR10DatasetAA(TensorDataset):
    """Generate adversarially crafted CIFAR10 data using AutoAttack library."""
    
    dset_name: str = "cifar10"
    dset_shape: tuple = (3, 32, 32)
    
    def __init__(
        self,
        model: torch.nn.Module,
        attack: str = None,  # e.g., 'apgd-ce', 'apgd-t'
        norm: str = 'Linf',
        eps: float = 8/255,
        version: str = 'standard',
        data_dir: str = osp.join(".", "data", "datasets"),
        checkpoint_fname: str = "autoattack_data.pt",
        adversarial_ratio: float = 1.0,
        small_magnitude_first: bool = False,
        verbose: bool = False,
        cache: bool = False,
        batch_size: int = 64,
    ):
        # Load CIFAR-10 test data
        _, ts = CDataLoaderCIFAR10().load(val_size=0)
        X, Y = ts.X / 255.0, ts.Y  # Normalize to [0,1] as required by AutoAttack
        
        # Convert to torch tensors in NCHW format
        X_tensor = carray2tensor(X, torch.float32).reshape(-1, 3, 32, 32)
        Y_tensor = carray2tensor(Y, torch.long)
        
        self.model = model
        self.device = next(model.parameters()).device
        
        fname = osp.join(data_dir, checkpoint_fname)
        if cache and osp.exists(fname):
            if verbose:
                print(f"Loaded adversarial CIFAR-10 dataset from {fname}")
            adv_X = torch.load(fname, map_location=self.device)
        else:
            if verbose:
                print("AutoAttack started...")
            
            # Move data to model's device
            X_tensor = X_tensor.to(self.device)
            Y_tensor = Y_tensor.to(self.device)
            
            # Create forward pass function for AutoAttack
            def forward_pass(x):
                # self.model.eval()
                # with torch.no_grad():
                #     return self.model(x)
                
                return self.model(x)
            
            # Initialize AutoAttack
            adversary = AutoAttack(
                forward_pass, 
                norm=norm, 
                eps=eps, 
                version=version, 
                verbose=verbose,
                device=self.device
            )
            if attack is not None: adversary.attacks_to_run = [attack]
            
            # Generate adversarial examples
            adv_X = adversary.run_standard_evaluation(X_tensor, Y_tensor, bs=batch_size)
            if verbose:
                print(f"AutoAttack complete! Dataset stored in {fname}")
            
            if cache:
                os.makedirs(data_dir, exist_ok=True)
                torch.save(adv_X.cpu(), fname)
        
        # Apply adversarial ratio
        if adversarial_ratio == 0.0:
            adv_X = X_tensor
        elif adversarial_ratio != 1.0:
            X_tensor = X_tensor.to(adv_X.device)
            dset_size = X_tensor.shape[0]
            split = int(adversarial_ratio * dset_size)

            if small_magnitude_first is True:
                # Get indices of samples with smallest perturbations
                attack_norms = (adv_X - X_tensor).norm(p=float("inf"), dim=(1,2,3))
                _, unpoison_ids = attack_norms.topk(dset_size - split, largest=False)
            else:
                # Random selection
                unpoison_ids = torch.randperm(dset_size)[:(dset_size - split)]

            adv_X[unpoison_ids] = X_tensor[unpoison_ids]
        
        # Move to CPU for dataset storage
        adv_X = adv_X.cpu()
        Y_tensor = Y_tensor.cpu()
        super().__init__(adv_X, Y_tensor)

