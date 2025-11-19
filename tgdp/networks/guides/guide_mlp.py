"""MLP Network for Guide Models."""

from copy import deepcopy
from typing import List, Optional

import torch
import torch.nn as nn
from tensordict import TensorDict

from ..embeddings import BaseEmbedding
from .base_guide_net import BaseGuideNet


class GuideMLPNet(BaseGuideNet):
    """MLP Network for Guide Models."""

    def __init__(
        self,
        input_size: int,
        sigma_embedding: Optional[BaseEmbedding] = None,
        condition_embedding: Optional[BaseEmbedding] = None,
        input_embedding: Optional[BaseEmbedding] = None,
        fully_connected_sizes: List[int] = [64, 128, 64],
        activation: nn.Module = nn.Mish(),
        norm_type: Optional[str] = None,
        use_dropout: bool = False,
    ):
        """Initialize the MLP.

        Args:
            input_size (int): Total input size for the network.
            sigma_embedding (Optional[BaseEmbedding]): Embedding for the sigma (noise level).
            condition_embedding (Optional[BaseEmbedding]): Optional embedding for conditioning information.
            input_embedding (Optional[BaseEmbedding]): Optional embedding for input data (e.g., noisy sample).
            fully_connected_sizes (List[int]): Sizes of fully connected layers.
            activation (nn.Module): Activation function to use.
            norm (Optional[str]): Normalization layer to use [batch, layer, None].
            use_dropout (bool): Whether to use dropout.

        """
        super().__init__()

        # Sigma embedding.
        self.sigma_embedding = sigma_embedding
        in_dim = 0
        if sigma_embedding is not None:
            in_dim += sigma_embedding.embedding_size

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

        # Normalization layer.
        norm = {"batch": nn.BatchNorm1d, "layer": nn.LayerNorm, None: nn.Identity}[norm_type]

        # Create the network.
        layers = []
        all_dims = [in_dim] + fully_connected_sizes + [1]
        for i in range(len(all_dims) - 1):
            layers.append(nn.Linear(all_dims[i], all_dims[i + 1]))
            # The last layer does not have an activation, norm, or dropout function.
            layers.append(norm(all_dims[i + 1])) if i < len(all_dims) - 2 else None
            layers.append(deepcopy(activation)) if i < len(all_dims) - 2 else None
            layers.append(nn.Dropout(0.1)) if i < len(all_dims) - 2 and use_dropout else None
        self.network = nn.Sequential(*layers)

    def forward(self, noisy_sample: torch.Tensor, sigma: torch.Tensor, extra_inputs: TensorDict) -> torch.Tensor:
        """Forward pass of the guide MLP network.

        Args:
            noisy_sample: The input sample.
            sigma: The noise level.
            extra_inputs: Extra inputs for the network.

        Returns:
            torch.Tensor: Output of the model.

        """
        # Embeddings.
        network_inputs = []
        if sigma is not None and sigma.shape == torch.Size([]):
            sigma = sigma.expand(noisy_sample.shape[0], 1)
        if self.sigma_embedding is not None:
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
        x = self.network(x)
        return x
