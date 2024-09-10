import torch
from torch.nn import CrossEntropyLoss

class FloodingCrossEntropyLoss(CrossEntropyLoss):
    def __init__(self, flood_level: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flood_level = flood_level

    def on_before_zero_grad(self, trainer, pl_module, optimizer):
        """
        Called after the backward pass but before optimizer step.
        Adjusts the loss by applying flooding.
        """
        current_loss = trainer.fit_loop.running_loss.last().item()
        flooded_loss = abs(current_loss - self.flood_level) + self.flood_level

        trainer.fit_loop.running_loss.append(flooded_loss)