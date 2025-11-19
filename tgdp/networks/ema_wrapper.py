"""EMA Wrapper for all nn.Modules."""

import logging
from copy import deepcopy
from typing import Optional

from torch import nn

logger = logging.getLogger(__name__)


class EMANetworkWrapper(nn.Module):
    """EMA Wrapper for nn.Modules.

    This class is a wrapper for torch.nn.Module objects that implements an Exponential Moving Average (EMA) model.
    It hold the original and the EMA model. The forward method returns the original model during training and the EMA
    model during evaluation. Updating the EMA model is done externally.
    """

    def __init__(
        self,
        wrapped_network: nn.Module,
        ema_decay: Optional[float] = None,
        stop_ema_step: Optional[int] = None,
    ):
        """Initialize the EMANetworkWrapper.

        Args:
            wrapped_network (nn.Module): The original network to wrap.
            ema_decay (Optional[float]): The decay parameter for the EMA update. 0 < decay < 1.
                Lower decay gives more weight to recent values. If None, the global decay is used.
            stop_ema_step (Optional[int]): If provided, stops updating the EMA after this training step.

        """
        super().__init__()
        # Original network and EMA network
        self._original = wrapped_network
        self._ema = deepcopy(self._original)
        self._ema_decay = ema_decay
        self._stop_ema_step = stop_ema_step

        # The EMA model should not be trained
        for param in self._ema.parameters():
            param.requires_grad_(False)

        logger.debug(f"EMA model created for {wrapped_network.__class__.__name__}.")

    def forward(self, *args, **kwargs):
        """Forward the input through the original or EMA model depending on the training state."""
        return self._original(*args, **kwargs) if self.training else self._ema(*args, **kwargs)
