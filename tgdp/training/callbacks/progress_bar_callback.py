"""Callback for a simple TQDM progress bar during training."""

import logging

import lightning.pytorch as pl
from tqdm import tqdm

logger = logging.getLogger(__name__)


class SimpleTQDMProgressBarCallback(pl.Callback):
    """Create a simple TQDM progress bar for the training loop."""

    def on_train_epoch_start(self, trainer: pl.Trainer, *args, **kwargs):
        """Create the progress bar at the beginning of the epoch."""
        self.pbar = tqdm(
            desc=f"Training Epoch {trainer.current_epoch}",
            total=int(trainer.limit_train_batches // trainer.accumulate_grad_batches),
            leave=False,
            bar_format="{desc:<20}{percentage:3.0f}%|{bar}{r_bar}",
        )

    def on_before_optimizer_step(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs):
        """Update the progress bar after each training step."""
        self.pbar.update(1)

    def on_train_epoch_end(self, trainer, *args, **kwargs):
        """Close the progress bar at the end of the epoch."""
        self.pbar.close()
