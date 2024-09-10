from pametric.lightning.callbacks.metric import PA_Callback

from pytorch_lightning import Trainer, LightningModule, LightningDataModule
from copy import deepcopy

class DebugPA(PA_Callback):
    """
    Subclass of PA_Callback to debug/test different aspects of the metric and the callback call.
    """
    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        import ipdb; ipdb.set_trace()
        # Build gradient graph for the model
        test_model = deepcopy(pl_module.model)
        import ipdb; ipdb.set_trace()
        for param in test_model.parameters():
            param.requires_grad = True
        test_model.train()

        pa_dict = self.pa_metric(
            classifier=test_model,
            local_rank=trainer.local_rank,
            destroy_process_group = False
        )
