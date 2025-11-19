"""MLP Network for Diffusion Models."""

from copy import deepcopy
from typing import List, Optional

import torch
import torch.nn as nn
from tensordict import TensorDict

from ..embeddings import BaseEmbedding
from .base_diffusion_net import BaseDiffusionNet


class DiffusionMLPNet(BaseDiffusionNet):
    """MLP Network for Diffusion Models.

    This class implements a simple mlp neural network for inverse dynamics prediction in diffusion models.
    This is not intended to be used with temporal data but with data of shape (batch, features).
    """

    def __init__(
        self,
        input_size: int,
        sigma_embedding: BaseEmbedding,
        condition_embedding: Optional[BaseEmbedding] = None,
        input_embedding: Optional[BaseEmbedding] = None,
        fully_connected_sizes: List[int] = [64, 128, 64],
        activation: nn.Module = nn.Mish(),
        norm: Optional[nn.Module] = None,
        dropout: float = 0.0,
    ):
        """Initialize the DiffusionMLP.

        Args:
            input_size (int): Total input size for the network.
            sigma_embedding (BaseEmbedding): Embedding for the sigma (noise level).
            condition_embedding (Optional[BaseEmbedding]): Optional embedding for conditioning information.
            input_embedding (Optional[BaseEmbedding]): Optional embedding for input data (e.g., noisy sample).
            fully_connected_sizes (List[int]): Sizes of fully connected layers.
            activation (nn.Module): Activation function to use.
            norm (Optional[str]): Optional normalization layer to use. ['batch', 'layer', None]
            dropout (float): Dropout probability.

        """
        super().__init__()

        # Sigma embedding.
        self.sigma_embedding = sigma_embedding
        in_dim = sigma_embedding.embedding_size

        # Condition embedding.
        self.condition_embedding = condition_embedding
        if condition_embedding is not None:
            in_dim += condition_embedding.embedding_size

        # Input embedding.
        self.input_embedding = input_embedding
        if input_embedding is not None:
            in_dim += input_embedding.embedding_size
        else:
            in_dim += input_size

        # Normalization, Dropout.
        use_norm = norm is not None
        use_dropout = dropout > 0.0

        # Create the network.
        layers = []
        all_dims = [in_dim] + fully_connected_sizes + [input_size]
        for i in range(len(all_dims) - 1):
            layers.append(nn.Linear(all_dims[i], all_dims[i + 1]))
            layers.append(deepcopy(norm)) if use_norm else None
            layers.append(deepcopy(activation)) if i < len(
                all_dims
            ) - 2 else None  # The last layer does not have an activation function.
            layers.append(nn.Dropout(dropout)) if i < len(all_dims) - 2 and use_dropout else None
        self.network = nn.Sequential(*layers)

    def forward(self, noisy_sample: torch.Tensor, sigma: torch.Tensor, extra_inputs: TensorDict) -> torch.Tensor:
        """Forward pass of the diffusion MLP network.

        Args:
            noisy_sample: The input sample.
            sigma: The noise level.
            extra_inputs: Extra inputs for the network.

        Returns:
            denoised_sample: The denoised sample.

        """
        assert noisy_sample.ndim == 2, "Input sample must be of shape (batch, features)."
        # Embeddings.
        network_inputs = []
        if sigma.shape == torch.Size([]):
            sigma = sigma.expand(noisy_sample.shape[0], 1)
        network_inputs.append(self.sigma_embedding(sigma))
        if self.condition_embedding is not None:
            network_inputs.append(self.condition_embedding(extra_inputs["global_condition"]))
        if self.input_embedding is not None:
            network_inputs.append(self.input_embedding(noisy_sample))
        else:
            network_inputs.append(noisy_sample)

        # Concatenate all embeddings.
        x = torch.cat(network_inputs, dim=-1)

        # Pass through the network.
        denoised_sample = self.network(x)
        return denoised_sample
