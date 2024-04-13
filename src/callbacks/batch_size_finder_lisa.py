from typing import Optional
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import BatchSizeFinder
class LISABatchSizeFinder(BatchSizeFinder):
    """
    Particular case of BatchSizeFinder that avoids OOM errors during LISA training.

    Detailed explanation: 
    
    Since LISA pairs observations with different labels/environments, the batch_size_OOM given by the standard
    BatchSizeFinder is subject to the strategy and the arbitrary pairing resulting from the first `steps_per_trial` steps.
    In order to guarantee that the batch size is never too large for the GPU, we need to consider the number of environments
    and impose a perfect pairing as an upper bound constraint. 

    We will define an attribute in the LightningModule that will make LISA aware of the OOM trial and just 
    concatenate the environment batches (i.e. return a perfect pairing) instead of performing the usual 
    selective augmentation pairing.
    """

    def __init__(self, percent_max_size: Optional[float] = 0.9,**kwargs):
        """
        Args:
            percent_max_size (Optional[float]): Percent of the maximum batch size found by the binsearch algorithm
                (if applicable) that will be used as an effective batch size. It should be lower than 1.0, as there is no
                guarantee that additional parameters stored in the GPU won't cause an OOM error.
        """
        super().__init__(**kwargs)

        # We override _steps_per_trial to be only 1, since we will generate an artificial batch with perfect pairing.
        self._steps_per_trial = 1
        self._percent_max_size = percent_max_size

    def scale_batch_size(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """
            Generate an artificial attribute in the pl_module that will be used to signal the OOM trial.
        """
        pl_module._BSF_oom_trial = True
        super().scale_batch_size(trainer, pl_module)
        pl_module._BSF_oom_trial = False

        self.optimal_batch_size = int(self.optimal_batch_size*self._percent_max_size)