"""Base-Class for MLP Networks."""

import torch
import torch.nn as nn


class BaseMLPNet(nn.Module):
    """Base class for Multi-Layer Perceptron (MLP) networks.

    This class serves as a template for MLP architectures. These are inteded as simple netwoeks that are not noise-
    conditonal.

    """

    def __init__(self):
        """Initialize the BaseMLPNet."""
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP network.

        Args:
            x: The input tensor.

        Returns:
            torch.Tensor: The output tensor after processing.

        """
        raise NotImplementedError("Forward pass not implemented.")
