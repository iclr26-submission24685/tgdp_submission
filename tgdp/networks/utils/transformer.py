"""Utility modules for building transformer networks.

Adapted from Dong et. al. - CleanDiffuser
Adapted from https://github.com/CleanDiffuserTeam/CleanDiffuser/blob/8cc60d410e365e04cb3838e7c42e0c45f72e85b1/cleandiffuser/nn_diffusion/dit.py
"""

#

import torch
import torch.nn as nn


def modulate(x, shift, scale):
    """Modulate the input tensor x using adaptive shift and scale using AdaLN as described in the DiT paper (variant 4).

    Args:
        x (torch.Tensor): Input tensor to be modulated.
        shift (torch.Tensor): Shift tensor for modulation.
        scale (torch.Tensor): Scale tensor for modulation.

    Returns:
        torch.Tensor: Modulated tensor.

    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning."""

    def __init__(self, hidden_size: int, n_heads: int, condition_size: int, dropout: float = 0.0):
        """Initialize the DiT block.

        This block consists of a multi-head self-attention layer and a feed-forward network,
        both of which are conditioned on an additional input (condition).
        The block uses adaptive layer normalization to modulate the input based on the condition.
        The attention and MLP blocks are gated by the condition, allowing for adaptive modulation.
        The block follows the design principles of DiT (Denoising Diffusion Transformer) as described in the paper.
        https://arxiv.org/abs/2203.03664.

        Args:
            hidden_size (int): Hidden size of the transformer.
            n_heads (int): Number of attention heads.
            condition_size (int): Size of the condition.
            dropout (float): Dropout rate.

        """
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, n_heads, dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        def approx_gelu():
            return nn.GELU(approximate="tanh")

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            approx_gelu(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )

        # We predict scale and shift based on the conditioning and adaptively modulate the LayerNorm.
        # In regular LayerNorm, the scale and shift are learned parameters (the affine parameters).
        # In addition we predict a gate for the attention and MLP blocks.
        # The gates are initialized to 0 so that the initial block is just an Identity function.
        # This is "variant 4" from the DiT paper.
        if condition_size != 0:
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(condition_size, hidden_size * 6))

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        """Forward pass of the block.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, horizon, hidden_size].
            cond (torch.Tensor): Condition tensor of shape [batch_size, condition_size].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, out_channels, horizon].

        """
        if cond is not None:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(cond).chunk(
                6, dim=1
            )  # 6 x (batch_size, hidden_size)
        else:
            # For no condition we set shift and scale to zero to disable modulation and gate to 1 to disable gating.
            shift_msa, scale_msa, shift_mlp, scale_mlp = torch.zeros(4, x.shape[0], x.shape[2], device=x.device)
            gate_msa, gate_mlp = torch.ones(2, x.shape[0], 1, device=x.device)
        x = modulate(self.norm1(x), shift_msa, scale_msa)  # (batch, horizon, hidden_size)
        x = x + gate_msa.unsqueeze(1) * self.attn(x, x, x)[0]  # (batch, horizon, hidden_size)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )  # (batch, horizon, hidden_size)
        return x


class FinalLayer1d(nn.Module):
    """Final layer of the DiT model.

    This layer applies a final normalization and a linear transformation to the output of the transformer.
    The output is of shape [batch_size, out_channels, horizon]. It also applies adaptive layer normalization modulation.
    """

    def __init__(self, hidden_size: int, condition_size: int, out_size: int):
        """Initialize the final layer of the DiT model.

        Args:
            hidden_size (int): Hidden size of the transformer.
            condition_size (int): Size of the condition.
            out_size (int): Output size.

        """
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_size)
        if condition_size > 0:
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(condition_size, 2 * hidden_size))

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        """Forward pass of the final layer.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_channels, horizon].
            cond (torch.Tensor): Condition tensor of shape [batch_size, condition_size].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, out_channels, horizon].

        """
        if cond is not None:
            shift, scale = self.adaLN_modulation(cond).chunk(2, dim=1)  # 2 x (batch_size, hidden_size)
        else:
            shift, scale = torch.zeros(2, x.shape[0], x.shape[2], device=x.device)
        x = modulate(self.norm_final(x), shift, scale)  # (batch, horizon, hidden_size)
        return self.linear(x)  # (batch, horizon, out_size)
