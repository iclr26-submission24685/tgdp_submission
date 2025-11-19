"""Base class for diffusion networks."""

from abc import ABC, abstractmethod
from typing import Optional

import torch
from tensordict import TensorDict


class BaseDiffusionNet(torch.nn.Module, ABC):
    """Base class for diffusion networks.

    This abstract class defines the interface for diffusion network models,
    requiring implementation of the forward method.
    """

    @abstractmethod
    def forward(
        self, noisy_sample: torch.Tensor, sigma: torch.Tensor, extra_inputs: Optional[TensorDict] = None
    ) -> torch.Tensor:
        """Perform a forward pass of the diffusion network.

        Args:
            noisy_sample (torch.Tensor): The input tensor containing the noisy sample.
            sigma (torch.Tensor): The noise level or standard deviation tensor.
            extra_inputs (Optional[TensorDict], optional): Additional inputs for the network.

        Returns:
            torch.Tensor: The output tensor after processing.

        """
        raise NotImplementedError
