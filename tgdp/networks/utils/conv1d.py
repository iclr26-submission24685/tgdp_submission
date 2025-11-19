"""Utility modules for building convolutional networks.

Adapted from Scheikl et. al. - Movement Primitive Diffusion
https://github.com/ScheiklP/movement-primitive-diffusion/blob/main/movement_primitive_diffusion/networks/conv1d.py
"""

import torch
from einops.layers.torch import Rearrange


class Conv_and_Sum(torch.nn.Module):
    """Computes a 1D convolution with width 1 and sums over the last dimension and applies a Mish non-linearity."""

    def __init__(self, in_channels: int):
        """Initialize the Conv-and-Sum layer.

        Args:
            in_channels (int): The number of input and output channels for the Conv1D layer.

        """
        super().__init__()
        self.conv = torch.nn.Conv1d(in_channels, in_channels, 1)
        self.mish = torch.nn.Mish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the conv and sum layer.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_channels, sequence_length].

        Returns:
            torch.Tensor: Output tensor. [batch_size, in_channels]

        """
        return self.mish(torch.sum(self.conv(x), dim=-1))


class SwapAxes(torch.nn.Module):
    """Swaps the two given dimensions of a tensor."""

    def __init__(self, dim1: int, dim2: int):
        """Initialize the SwapAxes layer.

        Args:
            dim1 (int): The first dimension to swap.
            dim2 (int): The second dimension to swap.

        """
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the SwapAxes layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with swapped dimensions.

        """
        return x.swapaxes(self.dim1, self.dim2)


# Adapted from https://github.com/columbia-ai-robotics/diffusion_policy/blob/main/diffusion_policy/model/diffusion/conv1d_components.py
class Downsample1d(torch.nn.Module):
    """Downsampling Layer.

    nn.Conv1d layer with default values that leave the channel dimension equal but downsamples (halves) the last
    dimension. It does so by using a kernel size of 3, stride of 2, and padding of 1.
    """

    def __init__(self, in_channels: int, kernel_size: int = 3, stride: int = 2, padding: int = 1):
        """Initialize the Downsample layer with the specified parameters.

        This layer is used to downsample the time dimension of the input of shape [batch, time, features].
        To halve the size of the input tensor, the default values should be used.

        Args:
            in_channels (int): Number of channels in the input signal.
            kernel_size (int, optional): Size of the convolving kernel. Defaults to 3.
            stride (int, optional): Stride of the convolution. Defaults to 2.
            padding (int, optional): Zero-padding added to both sides of the input. Defaults to 1.

        """
        super().__init__()
        self.conv = torch.nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the 1D convolution to the input tensor and returns the downsampled tensor.

        Args:
            x (torch.Tensor): The input tensor. [batch_size, in_channels, sequence_length]

        Returns:
            torch.Tensor: The downsampled tensor. [batch_size, in_channels, max(1, sequence_length // 2)]

        """
        return self.conv(x)


# Adapted from https://github.com/columbia-ai-robotics/diffusion_policy/blob/main/diffusion_policy/model/diffusion/conv1d_components.py
class Upsample1d(torch.nn.Module):
    """Upsampling layer.

    A 1D upsampling layer to increase the length of the input along the time dimension using transposed convolution.
    Per default, it doubles the dimensionality of the last dimension of the input tensor.
    """

    def __init__(self, in_channels: int, kernel_size: int = 4, stride: int = 2, padding: int = 1):
        """Initialize the Upsample1d layer with the specified parameters.

        Args:
            in_channels (int): Number of channels in the input signal.
            kernel_size (int, optional): Size of the convolving kernel. Default is 4.
            stride (int, optional): Stride of the convolution. Default is 2.
            padding (int, optional): Zero-padding added to both sides of the input. Default is 1.

        """
        super().__init__()
        self.conv = torch.nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass of the input tensor.

        Args:
            x (torch.Tensor): The input tensor. [batch_size, in_channels, sequence_length]

        Returns:
            torch.Tensor: The output tensor. [batch_size, in_channels, sequence_length * 2]

        """
        return self.conv(x)


# Adapted from https://github.com/columbia-ai-robotics/diffusion_policy/blob/main/diffusion_policy/model/diffusion/conv1d_components.py
class Conv1dBlock(torch.nn.Module):
    """Convolutional 1D Block. This applies: Conv1d --> Norm --> Activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        norm: str = "group",  # [group, layer, None]
        n_groups: int = 8,
    ):
        """Initialize the Conv1D block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel.
            norm (str): Normalization to use. Options are None, 'group' or 'layer'. Defaults to 'group'.
            n_groups (int, optional): Number of groups for GroupNorm (relevant if norm is 'group'). Defaults to 8.

        """
        super().__init__()

        # Define the normalization layer based on the specified type.
        if norm == "group":
            norm_layer = torch.nn.GroupNorm(n_groups, out_channels)
        elif norm == "layer":
            norm_layer = torch.nn.Sequential(
                Rearrange("b c h -> b h c"),
                torch.nn.LayerNorm(out_channels),
                Rearrange("b h c -> b c h"),
            )
        elif norm is None:
            norm_layer = torch.nn.Identity()
        else:
            raise ValueError(f"Unknown normalization type: {norm}")

        # Define the convolutional block with the specified parameters.
        self.block = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            norm_layer,
            torch.nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the convolutional block.

        Args:
            x (torch.Tensor): Input tensor. (batch, in_channels, horizon)

        Returns:
            torch.Tensor: Output tensor after applying the convolutional block. (batch, out_channels, horizon)

        """
        return self.block(x)


# Adapted from https://github.com/columbia-ai-robotics/diffusion_policy/blob/main/diffusion_policy/model/diffusion/conditional_unet1d.py
class ConditionalResidualBlock1D(torch.nn.Module):
    """A 1D Conditional Residual Block with FiLM modulation.

    This module applies a series of 1D convolutional blocks with optional
    Feature-wise Linear Modulation (FiLM) based on a given condition. It
    supports residual connections and can predict per-channel scale and bias
    for the condition.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        condition_size: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        predict_scale_of_condition: bool = False,
    ):
        """Initialize the Conditional Residual Block.

        Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        condition_size (int): Size of the condition vector.
        kernel_size (int, optional): Size of the convolutional kernel. Default is 3.
        n_groups (int, optional): Number of groups for group normalization. Default is 8.
        predict_scale_of_condition (bool, optional): If True, predicts per-channel
            scale and bias for the condition. Default is False.

        """
        super().__init__()

        self.blocks = torch.nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # Predicts per-channel scale and bias.
        cond_channels = out_channels
        if predict_scale_of_condition:
            cond_channels = out_channels * 2
        self.predict_scale_of_condition = predict_scale_of_condition
        self.out_channels = out_channels
        if condition_size > 0:
            self.cond_encoder = torch.nn.Sequential(
                torch.nn.Mish(),
                torch.nn.Linear(condition_size, cond_channels),
            )

        # make sure dimensions compatible
        self.residual_conv = (
            torch.nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else torch.nn.Identity()
        )

    def forward(self, x, cond):
        """Forward pass of the block.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_channels, horizon].
            cond (torch.Tensor): Condition tensor of shape [batch_size, condition_size].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, out_channels, horizon].

        """
        out = self.blocks[0](x)
        if cond is not None:
            embed = self.cond_encoder(cond)
            embed = embed.unsqueeze(-1)
            if self.predict_scale_of_condition:
                # Split output of cond_encoder into scale and bias
                embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
                scale = embed[:, 0, ...]
                bias = embed[:, 1, ...]
                out = scale * out + bias
            else:
                out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out
