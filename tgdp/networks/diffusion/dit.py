"""Decision Transformer (DiT) for Diffusion Models.

Adapted from https://github.com/CleanDiffuserTeam/CleanDiffuser/blob/8cc60d410e365e04cb3838e7c42e0c45f72e85b1/cleandiffuser/nn_diffusion/dit.py
"""

from typing import Optional

import torch
import torch.nn as nn
from tensordict import TensorDict

from ..embeddings import BaseEmbedding, SinusoidalProjection
from ..utils.transformer import DiTBlock, FinalLayer1d
from .base_diffusion_net import BaseDiffusionNet


class DiT(BaseDiffusionNet):
    """Diffusion Transformer (DiT) model for 1D diffusion tasks.

    This class implements a transformer-based architecture for diffusion models, supporting sigma and global condition
    embeddings, positional encoding, and multiple transformer blocks for processing sequential data.
    """

    def __init__(
        self,
        input_size: int,
        sigma_embedding: BaseEmbedding,
        global_condition_embedding: BaseEmbedding,
        hidden_size: int = 384,
        n_heads: int = 6,
        depth: int = 12,
        dropout: float = 0.0,
    ):
        """Diffusion Transformer, 1D Transformer for Diffusion Models.

        Args:
            input_size (int): Input dimension.
            sigma_embedding (nn.Module): Sigma embedding module.
            global_condition_embedding (nn.Module): Global condition embedding module.
            hidden_size (int): Hidden size of the transformer.
            n_heads (int): Number of attention heads.
            depth (int): Number of transformer blocks.
            dropout (float): Dropout rate.

        """
        super().__init__()
        # Embed sigma and global conditions from extra_inputs.
        self.sigma_embedding = sigma_embedding
        condition_size = sigma_embedding.embedding_size
        if global_condition_embedding is not None:
            self.global_condition = global_condition_embedding
            assert self.sigma_embedding.embedding_size == self.global_condition.embedding_size, (
                "sigma_embedding and global_condition_embedding must have the same embedding size."
            )

        # Transformer Positional Encoding.
        self.positional_embedding = SinusoidalProjection(hidden_size)
        self.pos_emb_cache = None

        # Global feature (sigma and condition) embedding.
        self.global_feature_embedding = nn.Sequential(
            nn.Linear(condition_size, hidden_size), nn.Mish(), nn.Linear(hidden_size, hidden_size), nn.Mish()
        )

        # Input Projection.
        self.input_projection = nn.Linear(input_size, hidden_size)

        # Transformer Layers
        self.blocks = nn.ModuleList([DiTBlock(hidden_size, n_heads, hidden_size, dropout) for _ in range(depth)])

        # Final Layer.
        self.final_layer = FinalLayer1d(hidden_size, hidden_size, input_size)

        # Initialize weights. This is important for the model to learn properly.
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize the weights of all layers in the DiT model."""

        # Initialize Transformer Layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize input projection:
        for module in self.input_projection.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                torch.nn.init.constant_(module.bias, 0)
                # torch.nn.init.normal_(module.weight, std=0.02)

        # Initialize time step embedding MLP:
        for module in self.sigma_embedding.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                torch.nn.init.constant_(module.bias, 0)
                # torch.nn.init.normal_(module.weight, std=0.02)

        # Initialize global condition embedding MLP:
        if hasattr(self, "global_condition"):
            for module in self.global_condition.modules():
                if isinstance(module, nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    torch.nn.init.constant_(module.bias, 0)
                    # torch.nn.init.normal_(module.weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)  # type: ignore
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)  # type: ignore

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)  # type: ignore
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)  # type: ignore
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, noisy_sample: torch.Tensor, sigma: torch.Tensor, extra_inputs: Optional[TensorDict] = None):
        """Forward pass of the DiT model.

        Args:
            noisy_sample (torch.Tensor): Noisy sample input.
            sigma (torch.Tensor): Sigma value.
            extra_inputs (Optional[TensorDict]): Extra inputs for the model.

        Returns:
            torch.Tensor: Output of the model.

        """
        x = noisy_sample  # (batch, horizon, input_size)

        # Positional Embeddings
        if self.pos_emb_cache is None or self.pos_emb_cache.shape[0] != x.shape[1]:
            self.pos_emb_cache = self.positional_embedding(
                torch.arange(x.shape[1], device=x.device).unsqueeze(-1)
            ).detach()  # (horizon, hidden_size)

        # Embed input and add positional embeddings.
        x = self.input_projection(x) + self.pos_emb_cache[None,]  # (batch, horizon, hidden_size)

        # Embed sigma and global condition.
        if sigma.shape == torch.Size([]):
            sigma = sigma.expand(noisy_sample.shape[0], 1, 1)
        global_feature = self.sigma_embedding(sigma.squeeze(-1))  # (batch, sigma_embedding_size)

        # Global condition embedding.
        if (
            extra_inputs is not None
            and "global_condition" in extra_inputs
            and extra_inputs["global_condition"] is not None
        ):
            global_condition = self.global_condition(extra_inputs["global_condition"])  # (batch, global_condition_size)
            global_feature += global_condition  # (batch, condition_size)

        # Local condition embedding.
        if (
            extra_inputs is not None
            and "local_condition" in extra_inputs
            and extra_inputs["local_condition"] is not None
        ):
            raise NotImplementedError("Local condition embedding is not implemented in DiT.")

        # Feature (sigma and condition) embedding.
        global_feature = self.global_feature_embedding(global_feature)

        # Transformer blocks.
        for block in self.blocks:
            x = block(x, global_feature)  # (batch, horizon, hidden_size)

        # Final layer.
        return self.final_layer(x, global_feature)  # (batch, horizon, input_size)
