"""Additional torch activation functions."""

import torch


class BangBangTanh(torch.nn.Module):
    """A Tanh activation function with a multiplicative factor that effectively acts as differentiable sign."""

    def __init__(self, factor: float = 1000.0):
        """Initialize the BangBangTanh layer.

        Args:
            factor(float): Multiplicative factor inside the Tanh.

        """
        super().__init__()
        self.tanh = torch.nn.Tanh()
        self.factor = factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict the output."""
        return self.tanh(self.factor * x)


class LocalMax(torch.nn.MaxPool1d):
    """A layer that computes the local maxima of the input tensor."""

    def __init__(self, kernel_size: int):
        """Initialize the LocMax layer.

        Args:
            kernel_size (int): The size of the kernel for the local maxima computation.

        """
        super().__init__(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)


class AdaptiveLocalSoftMax(torch.nn.Module):
    """A layer that computes the local soft maxima of the input tensor.

    This computes soft local maxima of an input tensor along a time dimension. The input tensor should be of shape
    [batch, time, features]. It computes a local denominator of a softmax as a convolution with a Gaussian kernel.
    """

    def __init__(self, sigma: float, temperature: float):
        """Initialize the LocMax layer.

        Args:
            sigma (float): The size (standard deviation) of the kernel for the local maxima computation.
            temperature(float): Temperature parameter for the softmax calculation.

        """
        super().__init__()
        self.sigma = sigma
        self.temperature = temperature

    def _set_kernel(self, device):
        self.kernel_width = int(self.sigma * 3)
        x = torch.linspace(-self.kernel_width, self.kernel_width, 2 * self.kernel_width + 1)
        kernel = torch.exp(-(x**2) / (2 * self.sigma**2))
        self.kernel = kernel.view(1, 1, -1).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the LocalMax layer.

        Args:
            x (torch.Tensor): Input tensor. (batch, time)

        Returns:
            torch.Tensor: Output tensor with the local maxima. (batch, time)

        """
        if not hasattr(self, "kernel"):
            self._set_kernel(x.device)

        x = x - x.max(dim=-1, keepdim=True).values

        norm = torch.nn.functional.conv1d(
            torch.exp(self.temperature * x).unsqueeze(1) + 1e-8, self.kernel, padding=self.kernel_width
        ).squeeze(1)
        out = torch.exp(self.temperature * x) / norm
        return out
