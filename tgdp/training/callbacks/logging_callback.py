"""Callbacks for logging various statistics during training with PyTorch Lightning."""

import logging
from typing import cast

import lightning.pytorch as pl
import psutil
import pynvml
import torch
import wandb
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

from ...agent.base_agent import BaseAgent
from ..base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class GradientNormLoggerCallback(pl.Callback):
    """LightningCallback for logging the mean gradient norms of the model parameters."""

    def __init__(self, log_every_n_steps: int = 1000, depth: int = 3):
        """Initialize the GradientNormLogger.

        Args:
            log_every_n_steps (int): The number of training steps between logging the mean gradient norms.
            depth (int): The depth of the model to log the gradient norms. All values for child modules lower than this
                depth will be averaged over.

        """
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.depth = depth

    def on_after_backward(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Call after the backward pass."""
        # Ensure that the trainer and pl_module are of the correct types
        trainer = cast(BaseTrainer, trainer)
        pl_module = cast(BaseAgent, pl_module)

        if trainer.global_step % self.log_every_n_steps == 0:
            self.log_gradient_norms(trainer, pl_module)

    def log_gradient_norms(self, trainer: BaseTrainer, pl_module: BaseAgent):
        """Log the mean gradient norms of the model parameters."""

        def get_mean_gradient_norms(module, current_depth=0, prefix=""):
            """Recursively computes the mean gradient norms of the model parameters."""
            if current_depth == self.depth:
                norm = 0.0
                num = 0
                for name, param in module.named_parameters():
                    if param.requires_grad:
                        # Accumulate the squared norm of the gradients
                        norm += param.grad.abs().sum().item()
                        num += param.grad.numel()
                if num == 0:
                    # If no parameters were found, return None
                    return {prefix.rstrip("."): None}
                # Return the square root of the accumulated squared norms
                return {prefix.rstrip("."): norm / num}

            results = {}
            for name, child in module.named_children():
                child_prefix = f"{prefix}{name}."
                # Recursively compute gradient norms for child modules
                results.update(get_mean_gradient_norms(child, current_depth + 1, child_prefix))
            return results

        # Compute mean gradient norms for the entire model
        mean_gradient_norm_dict = get_mean_gradient_norms(pl_module)

        for name, grad_norm in mean_gradient_norm_dict.items():
            if grad_norm is not None and trainer.logger is not None:
                # Log the gradient norm to the experiment logger
                trainer.logger.log_metrics({f"mean_gradient_norm/{name}": grad_norm}, step=trainer.global_step)


class WeightHistogramLoggerCallback(pl.Callback):
    """LightningCallback for logging the histograms of the model weights."""

    def __init__(self, log_every_n_steps: int = 1000, depth: int = 3):
        """Initialize the WeightHistogramLogger.

        Args:
            log_every_n_steps (int): The number of training steps between logging the mean gradient norms.
            depth (int): The depth of the model to log the gradient norms. All values for child modules lower than this
                depth will be averaged over.

        """
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.depth = depth

    def on_before_optimizer_step(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs):
        """Call at the end of every training step (this might be multiple batches for gradient accumulation)."""
        trainer_cast = cast(BaseTrainer, trainer)
        pl_module_cast = cast(BaseAgent, pl_module)
        if trainer_cast.global_step % self.log_every_n_steps == 0:
            self.log_weight_histograms(trainer_cast, pl_module_cast)

    def log_weight_histograms(
        self,
        trainer: BaseTrainer,
        pl_module: BaseAgent,
    ):
        """Log the histograms of the model weights."""

        def get_weights(module, current_depth=0, prefix=""):
            """Recursively collects the model weights."""
            if current_depth == self.depth:
                weights = []
                for name, param in module.named_parameters():
                    if param.requires_grad:
                        weights.extend(param.flatten().detach().cpu().numpy())
                return {prefix.rstrip("."): weights}

            results = {}
            for name, child in module.named_children():
                child_prefix = f"{prefix}{name}."
                results.update(get_weights(child, current_depth + 1, child_prefix))
            return results

        weight_dict = get_weights(pl_module)

        if trainer.logger is None:
            for name, weights in weight_dict.items():
                if weights:  # Check if the list is not empty
                    if isinstance(trainer.logger, TensorBoardLogger):
                        # Log the histogram to TensorBoard
                        trainer.logger.experiment.add_histogram(
                            f"weights/{name}", torch.tensor(weights), global_step=trainer.global_step
                        )
                    elif isinstance(trainer.logger, WandbLogger):
                        table = wandb.Table(data=[weights], columns=["weights"])
                        wandb.log({"episodes/episode_stats": table}, step=trainer.global_step)


class BasicStatsMonitorCallback(pl.Callback):
    """LightningCallback for monitoring basic statistics (CPU and GPU load and memory) of the training process."""

    def __init__(self, log_every_n_steps: int = 100):
        """Initialize the BasicStatsMonitor.

        Args:
            log_every_n_steps (int): The number of training steps between logging the basic statistics.

        """
        super().__init__()
        pynvml.nvmlInit()
        self.log_every_n_steps = log_every_n_steps
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assuming single GPU

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs):
        """Call at the end of every training batch."""
        # Note that this logs after every batch so for gradient accumulation, this may log more often then inteded.
        trainer_cast = cast(BaseTrainer, trainer)
        if trainer_cast.global_step % self.log_every_n_steps == 0:
            self.log_device_stats(trainer_cast)

    def log_device_stats(self, trainer: BaseTrainer):
        """Log the basic statistics of the training process."""
        # Log CPU usage
        cpu_usage = psutil.cpu_percent()

        # Log GPU utilization
        gpu_utilization = float(pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu)

        # Log memory usage
        memory = psutil.virtual_memory()
        memory_usage = memory.percent

        # Log GPU memory usage
        gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        gpu_memory_usage = float(gpu_memory.used) / float(gpu_memory.total) * 100

        # Log the metrics
        if trainer.logger is not None:
            trainer.logger.log_metrics(
                {
                    "device_stats/cpu_usage": cpu_usage,
                    "device_stats/gpu_utilization": gpu_utilization,
                    "device_stats/memory_usage": memory_usage,
                    "device_stats/gpu_memory_usage": gpu_memory_usage,
                },
                step=trainer.global_step,
            )

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Call at the end of training to clean up resources."""
        pynvml.nvmlShutdown()

