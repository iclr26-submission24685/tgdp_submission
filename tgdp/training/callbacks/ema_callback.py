"""LightningCallback for updating an Exponential Moving Average (EMA) models during training."""

import logging
from collections import OrderedDict
from typing import cast

import lightning.pytorch as pl
import torch

from ...networks.ema_wrapper import EMANetworkWrapper
from ..base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class UpdateEMACallback(pl.Callback):
    """LightningCallback for updating an Exponential Moving Average (EMA) agent during training."""

    def __init__(
        self,
        ema_decay: float,
        step_start_ema: int,
        update_ema_every: int,
        log_interval: int = 1000,
    ):
        """Initialize the UpdateEMACallback.

        Args:
            ema_agent (EMAAgent): The EMA agent to update. This has a original and an ema agent.
            ema_decay (float): The decay parameter for the EMA update. 0 < decay < 1.
                Lower decay gives more weight to recent values.
            step_start_ema (int): The training step (batch) number at which to start updating the EMA.
            update_ema_every (int): The number of training (batch) steps between EMA updates.
            log_interval (int): The number of training (batch) steps between EMA updates.

        """
        super().__init__()
        self.ema_decay = ema_decay
        self.step_start_ema = step_start_ema
        self.update_ema_every = update_ema_every
        self.log_interval = log_interval

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs):
        """Call at the beginning of training."""
        trainer = cast(BaseTrainer, trainer)  # Ensure the trainer is of type BaseTrainer
        # Find the EMA modules in the model
        self.ema_modules = self._find_ema_modules(pl_module)
        logger.debug(f"Found {len(self.ema_modules)} EMA modules in the agent.")

    def on_before_optimizer_step(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs):
        """Call at the end of every training step (this might be multiple batches for gradient accumulation)."""
        trainer = cast(BaseTrainer, trainer)  # Ensure the trainer is of type BaseTrainer
        if trainer.global_step % self.update_ema_every == 0:
            if trainer.global_step >= self.step_start_ema:
                # Update the EMA. This also does logging of the EMA difference.
                self.update_ema(trainer)
                logger.debug(f"EMA updated at step {trainer.global_step}")
            else:
                self.copy_original_to_ema()
                logger.debug(f"EMA copied from original at step {trainer.global_step}")

    def _find_ema_modules(self, module):
        ema_modules = []
        for child in module.children():
            if isinstance(child, EMANetworkWrapper):
                ema_modules.append(child)
            else:
                ema_modules.extend(self._find_ema_modules(child))
        return ema_modules

    @torch.no_grad()
    def update_ema(
        self,
        trainer: BaseTrainer,
    ):
        """Update the ema model with the original model's parameters buffers.

        This method updates Parameters and Buffers. Parameters are updated according to the EMA formula:
        `ema_param = ema_param - (1.0 - ema_decay) * (ema_param - original_param)`. Buffers are copied from the original
        to the ema model. This method should only be called during training.

        Args:
            original (torch.nn.Module): The original model.
            ema (torch.nn.Module): The EMA model.
            trainer (BaseTrainer): The trainer object for logging.
            log_stats (bool): Whether to log the mean parameter difference between the original and EMA models.

        """
        log = self.log_interval is not None and trainer.global_step % self.log_interval == 0
        # Keep track of the parameter difference for logging
        param_diff_sum = 0.0
        param_diff_count = 0

        # Update the EMA for each EMA module
        for module in self.ema_modules:
            # Set decay to the configured value.
            ema_decay = module._ema_decay if module._ema_decay is not None else self.ema_decay
            if module._stop_ema_step is not None and trainer.global_step >= module._stop_ema_step:
                continue # Skip EMA update if we reached the stop step

            # Update the EMA parameters
            for p_ema, p in zip(module._ema.parameters(), module._original.parameters()):
                p_ema.data.mul_(ema_decay).add_(p.data, alpha=1.0 - ema_decay)

                # Add the parameter difference to the sum.
                if log:
                    param_diff_sum += torch.sum(torch.abs(p - p_ema)).item()
                    param_diff_count += torch.numel(p)

            # Log the mean parameter difference
            if log and trainer.logger is not None and param_diff_count > 0:
                trainer.logger.log_metrics(
                    {"ema/mean_parameter_difference": param_diff_sum / param_diff_count},
                    step=trainer.global_step,
                )

            # Update the EMA buffers
            for buffer_ema, buffer in zip(module._ema.buffers(), module._original.buffers()):
                buffer_ema.data.copy_(buffer.data)

    @torch.no_grad()
    def copy_original_to_ema(self):
        """Copy the original model's parameters and buffers to the EMA model."""
        for module in self.ema_modules:
            # Copy the parameters
            original_params = OrderedDict(module._original.named_parameters())
            ema_params = OrderedDict(module._ema.named_parameters())
            assert original_params.keys() == ema_params.keys()
            for param_name, param in original_params.items():
                ema_params[param_name].copy_(param)

            # Copy the buffers
            original_buffers = OrderedDict(module._original.named_buffers())
            ema_buffers = OrderedDict(module._ema.named_buffers())
            assert original_buffers.keys() == ema_buffers.keys()
            for buffer_name, buffer in original_buffers.items():
                ema_buffers[buffer_name].copy_(buffer)
