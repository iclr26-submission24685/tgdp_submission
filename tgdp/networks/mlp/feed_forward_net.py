"""Basic feed-forward neural network for inverse dynamics models."""

from copy import deepcopy
from typing import List, Optional

import torch
from torch import nn

from .base_mlp_net import BaseMLPNet


class FeedForwardNet(BaseMLPNet):
    """A feed-forward inverse dynamics policy network."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        fully_connected_sizes: List[int] = [64, 128, 64],
        activation: nn.Module = nn.Mish(),
        norm: Optional[nn.Module] = None,
        use_dropout: bool = True,
        use_residual_connection: bool = True,
        out_activation: nn.Module = nn.Identity(),
    ):
        """Initialize the feed-forward inverse dynamics policy network.

        Args:
            input_size: The dimension of the input space.
            output_size: The dimension of the output space.
            fully_connected_sizes: The dimensions of the hidden fully connected layers.
            activation: The activation function of the hidden layers.
            norm: The normalization layer to use.
            use_dropout: Whether to use dropout.
            use_residual_connection: Whether to use a residual connection.
            out_activation: The activation function of the final layer.

        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        # Create the network.
        layers = []
        all_dims = [input_size] + fully_connected_sizes + [output_size]
        for i in range(len(all_dims) - 1):
            layers.append(nn.Linear(all_dims[i], all_dims[i + 1]))
            layers.append(deepcopy(norm)) if norm is not None else None
            layers.append(deepcopy(activation)) if i < len(
                all_dims
            ) - 2 else None  # The last layer does not have an activation function.
            layers.append(nn.Dropout(0.1)) if i < len(all_dims) - 2 and use_dropout else None
        if out_activation is not None:
            layers.append(out_activation)
        self.network = nn.Sequential(*layers)

        # Create the residual connection.
        self.use_residual_connection = use_residual_connection
        if use_residual_connection:
            self.residual = nn.Sequential(nn.Linear(input_size, output_size))

    def forward(
        self,
        x: torch.Tensor,
    ):
        """Predict the output given the input.

        Args:
            x: The input tensor.
            future: The observation at time t+1.

        Returns:
            output: The predicted output.

        """
        if self.use_residual_connection:
            x = self.network(x) + self.residual(x)
        else:
            x = self.network(x)
        return x
