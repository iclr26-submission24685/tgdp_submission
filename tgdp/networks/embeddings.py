# Adapted from Scheikl et. al. - Movement Primitive Diffusion
"""A collection of embedding classes for use in neural network architectures.

Features embeddings for the noiselevel, position, and conditions.

Adapted from Scheikl et. al. - Movement Primitive Diffusion
https://github.com/ScheiklP/movement-primitive-diffusion/blob/main/movement_primitive_diffusion/networks/sigma_embeddings.py
"""

import math
from copy import deepcopy

import torch


class BaseEmbedding(torch.nn.Module):
    """Base class for sigma/position/condition embeddings. These may be static or learning."""

    embedding_size: int  # output size of the embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embeds the input.

        Args:
            x: Input tensor containing input values. (batch, 1)

        Returns:
            torch.Tensor: Output tensor after applying the embedding. (batch, embedding_size)

        """
        raise NotImplementedError("This method should be implemented by subclasses.")


class LinearEmbedding(BaseEmbedding):
    """Linear embedding.

    Embeds the input with a single linear layer with learnable weights.

    Args:
        embedding_size: Size of the embedding.

    """

    def __init__(self, embedding_size: int):
        """Initialize the LinearEmbedding with the specified embedding size.

        Args:
            embedding_size (int): Size of the embedding.

        """
        super().__init__()
        self.embedding_size = embedding_size
        self.embed = torch.nn.Linear(1, embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embeds the input tensor using a linear layer.

        Args:
            x (torch.Tensor): Input tensor to be embedded.

        Returns:
            torch.Tensor: Output tensor after applying the linear embedding.

        """
        return self.embed(x)


class RepeatEmbedding(BaseEmbedding):
    """Repeat embedding.

    Repeats the input embedding multiple times.

    Args:
        embedding_size: Size of the embedding.

    """

    def __init__(self, embedding_size: int):
        """Initialize the SinusoidalProjection with the specified embedding size.

        Args:
            embedding_size (int): Size of the embedding.

        """
        super().__init__()
        self.embedding_size = embedding_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Repeats the input tensor along the last dimension to match the embedding size.

        Args:
            x (torch.Tensor): Input tensor to be repeated.

        Returns:
            torch.Tensor: Output tensor with repeated values along the last dimension.

        """
        return x.expand(-1, self.embedding_size)


class PassThroughEmbedding(BaseEmbedding):
    """Embedding that passes the input through unchanged.

    This embedding simply returns the input tensor as is, with an embedding size of 1.
    """

    def __init__(self, embedding_size: int = 1):
        """Initialize the PassThroughEmbedding with embedding size 1."""
        super().__init__()
        self.embedding_size = embedding_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the input tensor unchanged.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The same input tensor.

        """
        return x


class GaussianFourierProjection(BaseEmbedding):
    """Gaussian random features.

    Randomly sample weights during initialization.
    These weights are fixed during optimization and are not trainable.

    Args:
        embedding_size: Size of the embedding.
        std: Standard deviation of the Gaussian distribution.

    """

    def __init__(self, embedding_size: int, std: float = 30.0):
        """Initialize the GaussianFourierProjection with fixed random weights.

        Args:
            embedding_size (int): Size of the embedding.
            std (float, optional): Standard deviation of the Gaussian distribution. Defaults to 30.0.

        """
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed during optimization.
        self.W = torch.nn.Parameter(torch.randn(embedding_size // 2) * std, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the Gaussian Fourier projection to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, 1).

        Returns:
            torch.Tensor: Output tensor after applying the projection, shape (batch, embedding_size).

        """
        x_proj = x[:, None] * self.W[None, :] * 2 * torch.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class GaussianFourierEmbedding(BaseEmbedding):
    """Gaussian random features embedding.

    Input is embedded using Gaussian random features, and then passed through a
    linear layer, a non-linearity, and another linear layer.

    Args:
        embedding_size: Size of the embedding.
        std: Standard deviation of the Gaussian distribution.

    """

    def __init__(self, embedding_size: int, std: float = 30.0):
        """Initialize the GaussianFourierEmbedding with the specified embedding size and standard deviation.

        Args:
            embedding_size (int): Size of the embedding.
            std (float, optional): Standard deviation of the Gaussian distribution. Defaults to 30.0.

        """
        super().__init__()
        self.embedding_size = embedding_size
        self.embed = torch.nn.Sequential(
            GaussianFourierProjection(embedding_size=embedding_size, std=std),
            torch.nn.Linear(embedding_size, 2 * embedding_size),
            torch.nn.Mish(),
            torch.nn.Linear(2 * embedding_size, embedding_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the GaussianFourierEmbedding.

        Args:
            x (torch.Tensor): Input tensor to be embedded.

        Returns:
            torch.Tensor: Output tensor after applying the Gaussian Fourier embedding.

        """
        return self.embed(x.squeeze(-1))


class FourierFeaturesProjection(BaseEmbedding):
    """Fourier features projection.

    Input is projected using Fourier features with random weights.

    Args:
        embedding_size: Size of the embedding.
        in_features: Number of input features.
        std: Standard deviation of the Gaussian distribution.

    """

    # Weight for the Fourier features embedding, initialized with random values.
    weight: torch.Tensor

    def __init__(self, embedding_size: int, std: float = 16.0):
        """Initialize the FourierFeaturesEmbedding with the specified embedding size and standard deviation.

        Args:
            embedding_size (int): Size of the embedding. Must be even.
            std (float, optional): Standard deviation of the Gaussian distribution for weight init. Defaults to 1.0.

        """
        super().__init__()
        assert embedding_size % 2 == 0, "Embedding size must be even."
        self.register_buffer("weight", torch.randn([embedding_size // 2, 1]) * std)
        self.embedding_size = embedding_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed the input tensor using Fourier features.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, 1) or (batch,).

        Returns:
            torch.Tensor: Output tensor after applying the Fourier features embedding, shape (batch, embedding_size).

        """
        if len(x.shape) == 1:
            x = x.unsqueeze(1)
        f = 2 * torch.pi * x @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


class FourierFeaturesEmbedding(BaseEmbedding):
    """Fourier features embedding.

    Input is embedded using Fourier features, and then passed through a
    linear layer, a non-linearity, and another linear layer.

    Args:
        embedding_size: Size of the embedding.
        std: Standard deviation of the Gaussian distribution.

    """

    def __init__(self, embedding_size: int, std: float = 16.0):
        """Initialize the FourierFeaturesEmbedding with the specified embedding size and standard deviation.

        Args:
            embedding_size (int): Size of the embedding. Must be even.
            std (float, optional): Standard deviation of the Gaussian distribution for weight init. Defaults to 1.0.

        """
        super().__init__()
        self.embedding_size = embedding_size
        self.embed = torch.nn.Sequential(
            FourierFeaturesProjection(embedding_size=embedding_size // 4, std=std),
            torch.nn.Linear(embedding_size // 4, embedding_size),
            torch.nn.Mish(),
            torch.nn.Linear(embedding_size, embedding_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the FourierFeaturesEmbedding.

        Args:
            x (torch.Tensor): Input tensor to be embedded.

        Returns:
            torch.Tensor: Output tensor after applying the Fourier features embedding.

        """
        return self.embed(x.squeeze(-1))


class MLPEmbedding(BaseEmbedding):
    """MLP embedding.

    Input is embedded using a multi-layer perceptron. Input size defaults to 1, but can be set to any value.
    Consists of a linear layer, a non-linearity, and another linear layer.

    Args:
        embedding_size: Size of the embedding.

    """

    def __init__(
        self,
        embedding_size: int,
        input_size: int = 1,
        hidden_size_factor: int = 2,
        hidden_layers: int = 1,
        activation_fn: torch.nn.Module = torch.nn.Mish(),
    ):
        """Initialize the MLP Embeddings module.

        Args:
            embedding_size (int): The size of the embedding vector.
            input_size (int, optional): The size of the input vector. Defaults to 1.
            hidden_size_factor (int, optional): The factor by which the hidden layer size is increased. Defaults to 2.
            hidden_layers (int, optional): The number of hidden layers with size embedding_size*hidden_size_factor.
                If no hidden layer is used, the NN is [1->embedding_size->embedding_size]. Technically this is
                hidden_layers+1. Defaults to 1.
            activation_fn (torch.nn.Module, optional): The activation function to be used. Defaults to torch.nn.Mish().

        """
        super().__init__()
        self.embedding_size = embedding_size
        self.embed = torch.nn.Sequential(
            torch.nn.Linear(input_size, embedding_size),
            deepcopy(activation_fn),
            *[
                torch.nn.Sequential(
                    torch.nn.Linear(
                        embedding_size * (hidden_size_factor if int(i > 0) else 1), embedding_size * hidden_size_factor
                    ),
                    deepcopy(activation_fn),
                )
                for i in range(hidden_layers)
            ],
            torch.nn.Linear(embedding_size * (hidden_size_factor if int(hidden_layers > 0) else 1), embedding_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the embedding.

        Args:
            x (torch.Tensor): Input tensor containing x values.

        Returns:
            torch.Tensor: Output tensor after applying the embedding.

        """
        return self.embed(x)


# From https://github.com/CleanDiffuserTeam/CleanDiffuser/blob/main/cleandiffuser/utils/utils.py
class PositionalProjection(BaseEmbedding):
    """Positional embedding using sinusoidal functions.

    This embedding maps input positions to a higher-dimensional space using
    sinusoidal functions, similar to the positional encodings used in transformer models.

    Args:
        embedding_size (int): Size of the embedding (must be even).
        max_positions (int, optional): Maximum number of positions. Defaults to 10000.
        endpoint (bool, optional): Whether to use the endpoint in frequency calculation. Defaults to False.

    """

    def __init__(self, embedding_size: int, max_positions: int = 10000, endpoint: bool = False):
        """Initialize the PositionalEmbedding.

        Args:
            embedding_size (int): Size of the embedding (must be even).
            max_positions (int, optional): Maximum number of positions. Defaults to 10000.
            endpoint (bool, optional): Whether to use the endpoint in frequency calculation. Defaults to False.

        """
        super().__init__()
        self.embedding_size = embedding_size
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        """Compute the positional embedding for the input tensor.

        Args:
            x (torch.Tensor): Input tensor of positions, shape (batch, 1).

        Returns:
            torch.Tensor: Positional embedding tensor, shape (batch, embedding_size).

        """
        freqs = torch.arange(start=0, end=self.embedding_size // 2, dtype=x.dtype, device=x.device)
        freqs = freqs / (self.embedding_size // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x * freqs
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class PositionalEmbedding(BaseEmbedding):
    """Positional embedding using sinusoidal functions.

    This embedding maps input positions to a higher-dimensional space using
    sinusoidal functions, similar to the positional encodings used in transformer models.
    It then applies a linear layer, a non-linearity, and another linear layer.

    Args:
        embedding_size (int): Size of the embedding (must be even).
        hidden_size_factor (int, optional): The factor by which the hidden layer size is increased. Defaults to 2.
        max_positions (int, optional): Maximum number of positions. Defaults to 10000.
        endpoint (bool, optional): Whether to use the endpoint in frequency calculation. Defaults to False.

    """

    def __init__(
        self, embedding_size: int, hidden_size_factor: int = 2, max_positions: int = 10000, endpoint: bool = False
    ):
        """Initialize the PositionalEmbedding.

        Args:
            embedding_size (int): Size of the embedding (must be even).
            hidden_size_factor (int, optional): The factor by which the hidden layer size is increased. Defaults to 2.
            max_positions (int, optional): Maximum number of positions. Defaults to 10000.
            endpoint (bool, optional): Whether to use the endpoint in frequency calculation. Defaults to False.

        """
        super().__init__()
        self.embedding_size = embedding_size
        self.embed = torch.nn.Sequential(
            PositionalProjection(embedding_size=embedding_size, max_positions=max_positions, endpoint=endpoint),
            torch.nn.Linear(embedding_size, hidden_size_factor * embedding_size),
            torch.nn.Mish(),
            torch.nn.Linear(hidden_size_factor * embedding_size, embedding_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the embedding.

        Args:
            x (torch.Tensor): Input tensor containing input values. (batch, 1)

        Returns:
            torch.Tensor: Output tensor after applying the embedding. (batch, embedding_size)

        """
        return self.embed(x)


class SinusoidalProjection(BaseEmbedding):
    """Sinusoidal projection.

    Input is embedded using sinusoidal projection.

    Args:
        embedding_size: Size of the embedding.

    """

    def __init__(self, embedding_size: int):
        """Initialize the SinusoidalProjection with the specified embedding size.

        Args:
            embedding_size (int): Size of the embedding.

        """
        super().__init__()
        self.embedding_size = embedding_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embeds the input using sinusoidal projection.

        Args:
            x: Input tensor containing input values. (batch, 1)

        Returns:
            torch.Tensor: Output tensor after applying the embedding. (batch, embedding_size)

        """
        device = x.device
        half_dim = self.embedding_size // 2
        embedding = math.log(10000) / (half_dim - 1)
        embedding = torch.exp(torch.arange(half_dim, device=device) * -embedding)
        embedding = x * embedding.unsqueeze(0)  # (batch, embedding_size/2)
        embedding = torch.cat((embedding.sin(), embedding.cos()), dim=-1)
        return embedding


class SinusoidalEmbedding(BaseEmbedding):
    """Sinusoidal embedding.

    Input is embedded using sinusoidal projection, and then passed through a
    linear layer, a non-linearity, and another linear layer.
    """

    def __init__(self, embedding_size: int, hidden_size_factor: int = 2):
        """Initialize the Sinusoidal Embeddings module.

        Args:
            embedding_size (int): The size of the embedding vector.
            hidden_size_factor (int, optional): The factor by which the hidden layer size is increased. Defaults to 2.

        """
        super().__init__()
        self.embedding_size = embedding_size
        self.embed = torch.nn.Sequential(
            SinusoidalProjection(embedding_size=embedding_size),
            torch.nn.Linear(embedding_size, embedding_size * hidden_size_factor),
            torch.nn.Mish(),
            torch.nn.Linear(embedding_size * hidden_size_factor, embedding_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the embedding.

        Args:
            x (torch.Tensor): Input tensor containing input values. (batch, 1)

        Returns:
            torch.Tensor: Output tensor after applying the embedding. (batch, embedding_size)

        """
        return self.embed(x)
