"""Half Diffusion Transformer (DiT) Classifier Network.

This network is designed for image classification tasks using a diffusion-based approach.
Adapted from https://github.com/CleanDiffuserTeam/CleanDiffuser/blob/8cc60d410e365e04cb3838e7c42e0c45f72e85b1/cleandiffuser/nn_diffusion/dit.py
"""

from typing import Optional

import torch
import torch.nn as nn
from tensordict import TensorDict

from ..embeddings import SinusoidalProjection
from ..utils.transformer import DiTBlock, FinalLayer1d
from .base_guide_net import BaseGuideNet


class HalfDiT(BaseGuideNet):
    """Half Diffusion Transformer (DiT) Classifier Network.

    This class implements a 1D Transformer-based architecture for diffusion models,
    designed for image classification tasks using diffusion-based approaches.

    """

    def __init__(
        self,
        input_size: int,
        out_size: int,
        sigma_embedding: nn.Module,
        global_condition_embedding: nn.Module,
        hidden_size: int = 384,
        n_heads: int = 6,
        depth: int = 12,
        dropout: float = 0.0,
        project_output_from_dit: bool = False,
    ):
        """Diffusion Transformer, 1D Transformer for Diffusion Models.

        Args:
            input_size (int): Input dimension.
            out_size (int): Output dimension.
            sigma_embedding (nn.Module): Sigma embedding module.
            global_condition_embedding (nn.Module): Global condition embedding module.
            hidden_size (int): Hidden size of the transformer.
            n_heads (int): Number of attention heads.
            depth (int): Number of transformer blocks.
            dropout (float): Dropout rate.
            project_output_from_dit (bool): Whether to project the output from the DiT. If false, we implement a simpler
                net that is used in the DV paper which uses a simple MLP after transformer blocks. If true, we use the
                more complex HalfDiT from CleanDiffuser, thatcomputes the DiT and then classify from using an MLP.

        """
        super().__init__()
        self.project_output_from_dit = project_output_from_dit

        # Embed sigma and global conditions from extra_inputs.
        condition_size = 0
        if sigma_embedding is not None:
            self.sigma_embedding = sigma_embedding
            condition_size = sigma_embedding.embedding_size
        if global_condition_embedding is not None:
            self.global_condition = global_condition_embedding
            if sigma_embedding is not None:
                assert self.sigma_embedding.embedding_size == self.global_condition.embedding_size, (
                    "sigma_embedding and global_condition_embedding must have the same embedding size."
                )
            else:
                condition_size = global_condition_embedding.embedding_size

        # Condition projection.
        if condition_size != 0:
            self.condition_projection = nn.Sequential(
                nn.Linear(condition_size, hidden_size), nn.Mish(), nn.Linear(hidden_size, hidden_size), nn.Mish()
            )

        # Transformer Positinal Embedding.
        self.positional_embedding = SinusoidalProjection(hidden_size)
        self.pos_emb_cache = None

        # Input Projection.
        self.input_projection = nn.Linear(input_size, hidden_size)

        # Transformer Layers
        self.blocks = nn.ModuleList([DiTBlock(hidden_size, n_heads, condition_size, dropout) for _ in range(depth)])

        # Final Layer and Projection.
        if not project_output_from_dit:
            self.final_layer = nn.Linear(hidden_size, out_size)

        else:
            self.final_layer = FinalLayer1d(hidden_size, condition_size, out_size=hidden_size // 2)

            # Output Projection.
            self.proj = nn.Sequential(
                nn.LayerNorm(hidden_size // 2),
                nn.SiLU(),
                nn.Linear(hidden_size // 2, hidden_size // 4),
                nn.LayerNorm(hidden_size // 4),
                nn.SiLU(),
                nn.Linear(hidden_size // 4, out_size),
            )

        # Initialize weights.
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize the weights of the transformer layers, embedding modules, and output layers."""

        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize time step embedding MLP:
        if hasattr(self, "sigma_embedding") and self.sigma_embedding is not None:
            for module in self.sigma_embedding.modules():
                if isinstance(module, nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    torch.nn.init.constant_(module.bias, 0)
                    # torch.nn.init.normal_(module.weight, std=0.02)

        # Initialize global condition embedding MLP:
        if hasattr(self, "global_condition") and self.global_condition is not None:
            for module in self.global_condition.modules():
                if isinstance(module, nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    torch.nn.init.constant_(module.bias, 0)
                    # torch.nn.init.normal_(module.weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            if hasattr(block, "adaLN_modulation") and block.adaLN_modulation is not None:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        if self.project_output_from_dit:
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(self.final_layer.linear.weight, 0)
            nn.init.constant_(self.final_layer.linear.bias, 0)
        else:
            nn.init.xavier_uniform_(self.final_layer.weight)
            nn.init.constant_(self.final_layer.bias, 0)

    def forward(
        self, noisy_sample: torch.Tensor, sigma: Optional[torch.Tensor], extra_inputs: Optional[TensorDict] = None
    ):
        """Forward pass of the DiT model.

        Args:
            noisy_sample (torch.Tensor): Noisy sample input.
            sigma (Optional[torch.Tensor]): Sigma value.
            extra_inputs (Optional[TensoDict]): Extra inputs for the model.

        Returns:
            torch.Tensor: Output of the model.

        """
        x = noisy_sample  # (batch, horizon, input_size)

        # Positional Embeddings.
        if self.pos_emb_cache is None or self.pos_emb_cache.shape[0] != x.shape[1]:
            self.pos_emb_cache = self.positional_embedding(
                torch.arange(x.shape[1], device=x.device).unsqueeze(-1)
            ).detach()  # (horizon, hidden_size)

        # Embed input and add positional embeddings.
        x = self.input_projection(x) + self.pos_emb_cache[None,]  # (batch, horizon, hidden_size)

        # Embed sigma.
        if sigma is not None:
            if sigma.shape == torch.Size([]):
                sigma = sigma.expand(noisy_sample.shape[0], 1, 1)
            global_feature = self.sigma_embedding(sigma.squeeze(-1))  # (batch, sigma_embedding_size)
        else:
            global_feature = None

        # Global condition embedding.
        if (
            extra_inputs is not None
            and "global_condition" in extra_inputs
            and extra_inputs["global_condition"] is not None
        ):
            global_condition = self.global_condition(extra_inputs["global_condition"])  # (batch, global_condition_size)
            if global_feature is not None:
                global_feature += global_condition  # (batch, sigma_embedding_size + global_condition_size)
            else:
                global_feature = global_condition

        # Local condition embedding.
        if (
            extra_inputs is not None
            and "local_condition" in extra_inputs
            and extra_inputs["local_condition"] is not None
        ):
            raise NotImplementedError("Local condition embedding is not implemented in DiT.")

        # Transformer blocks.
        for block in self.blocks:
            x = block(x, global_feature)  # (batch, horizon, hidden_size)

        # Final layer and output projection.
        if not self.project_output_from_dit:
            x = self.final_layer(x)  # (batch, horizon, out_size)
            return x[:, 0, :]  # (batch, out_size)
        else:
            x = self.final_layer(x, global_feature)  # (batch, horizon, input_size)
            x = x.mean(dim=1)  # (batch, hidden_size)
            return self.proj(x)  # (batch, out_size)
