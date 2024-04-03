from typing import Optional
from pytorch_lightning import Trainer, LightningModule
from pametric.lightning.callbacks import MultienvBatchSizeFinder

class LISABatchSizeFinder(MultienvBatchSizeFinder):
    """
    Particular case of MultienvBatchSizeFinder that avoids OOM errors during LISA training.

    Detailed explanation: 
    
    Since LISA pairs observations with different labels/environments, the batch_size_OOM given by the standard
    BatchSizeFinder is subject to the strategy and the arbitrary pairing resulting from the first `steps_per_trial` steps.
    In order to guarantee that the batch size is never too large for the GPU, we need to consider the number of environments
    and impose a perfect pairing as an upper bound constraint. 

    Most of the work has already implemented in the `pametric` callback, but we need to override some configurations and also define
    an attribute in the LightningModule that will make LISA aware of the OOM trial and just concatenate the environment batches (i.e.
    return a perfect pairing) instead of performing the usual selective augmentation pairing.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # We override _steps_per_trial to be only 1, since we will generate an artificial batch with perfect pairing.
        self._steps_per_trial = 1

    def scale_batch_size(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """
            Generate an artificial attribute in the pl_module that will be used to signal the OOM trial.
        """
        pl_module._BSF_oom_trial = True
        super().scale_batch_size(trainer, pl_module)
        pl_module._BSF_oom_trial = False

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule):
        """
        We will change the validation batch size because the environment structure might be different from training.
        """
        if trainer.sanity_checking or trainer.state.fn != "validate":
            return

        self.scale_batch_size(trainer, pl_module)