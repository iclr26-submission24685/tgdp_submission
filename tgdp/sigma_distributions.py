"""Sigma distributions for sampling noise levels in diffusion model training."""

# Adapted from Scheikl et. al. - Movement Primitive Diffusion
# https://github.com/ScheiklP/movement-primitive-diffusion/blob/main/movement_primitive_diffusion/distributions.py

import math
from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np
import torch


class BaseSigmaDistribution(ABC):
    """Abstract base class for sigma distributions used in diffusion model training."""

    def __init__(self):
        """Initialize the base sigma distribution."""
        super().__init__()

    @abstractmethod
    def sample(self, shape: Iterable[int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Sample noise levels (sigmas) according to the distribution.

        Args:
            shape (Iterable[int]): The shape of the tensor to sample.
            device (torch.device): The device to allocate the tensor on.
            dtype (torch.dtype): The data type of the tensor.

        Returns:
            torch.Tensor: A tensor of sampled sigma values.

        """
        raise NotImplementedError


class RandLogNormal(BaseSigmaDistribution):
    """A log-normal distribution for sampling noise levels in diffusion model training."""

    def __init__(self, loc=0.0, scale=1.0):
        """Initialize a log-normal distribution with given location and scale.

        Args:
            loc (float): The mean of the underlying normal distribution.
            scale (float): The standard deviation of the underlying normal distribution.

        """
        super().__init__()
        self.loc = loc
        self.scale = scale

    def sample(self, shape: Iterable[int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Draws samples from a lognormal distribution."""
        return (torch.randn(*shape, device=device, dtype=dtype) * self.scale + self.loc).exp()


class RandLogLogistic(BaseSigmaDistribution):
    """A log-logistic distribution with location `loc` and scale `scale`. This is used by Karras et. al."""

    def __init__(self, loc=1.0, scale=1.0, min_value=0.0, max_value=float("inf")):
        """Initialize a log-logistic distribution with location, scale, min, and max values.

        This distribution is used in Karras et. al.

        Args:
            loc (float): The location parameter of the distribution.
            scale (float): The scale parameter of the distribution.
            min_value (float): The minimum value for truncation.
            max_value (float): The maximum value for truncation.

        """
        super().__init__()
        self.loc = loc
        self.scale = scale
        self.min_value = min_value
        self.max_value = max_value

    def sample(self, shape: Iterable[int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Draws samples from an optionally truncated log-logistic distribution."""
        min_value = torch.as_tensor(self.min_value, device=device, dtype=torch.float64)
        max_value = torch.as_tensor(self.max_value, device=device, dtype=torch.float64)
        min_cdf = min_value.log().sub(self.loc).div(self.scale).sigmoid()
        max_cdf = max_value.log().sub(self.loc).div(self.scale).sigmoid()
        u = torch.rand(*shape, device=device, dtype=torch.float64) * (max_cdf - min_cdf) + min_cdf
        return u.logit().mul(self.scale).add(self.loc).exp().to(dtype)


class RandLogUniform(BaseSigmaDistribution):
    """A log-uniform distribution for sampling noise levels in diffusion model training."""

    def __init__(self, min_value, max_value):
        """Initialize a log-uniform distribution with minimum and maximum values.

        Args:
            min_value (float): The minimum value of the distribution.
            max_value (float): The maximum value of the distribution.

        """
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value

    def sample(self, shape: Iterable[int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Draws samples from a log-uniform distribution."""
        min_value = math.log(self.min_value)
        max_value = math.log(self.max_value)
        return (torch.rand(*shape, device=device, dtype=dtype) * (max_value - min_value) + min_value).exp()


class RandNormal(BaseSigmaDistribution):
    """A normal (Gaussian) distribution for sampling noise levels in diffusion model training."""

    def __init__(self, loc=0.0, scale=1.0):
        """Initialize a normal (Gaussian) distribution with given location and scale.

        Args:
            loc (float): The mean of the normal distribution.
            scale (float): The standard deviation of the normal distribution.

        """
        super().__init__()
        self.loc = loc
        self.scale = scale

    def sample(self, shape: Iterable[int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Draws samples from a normal distribution."""
        return torch.randn(*shape, device=device, dtype=dtype) * self.scale + self.loc


class RandUniform(BaseSigmaDistribution):
    """A uniform distribution for sampling noise levels in diffusion model training."""

    def __init__(self, min_value, max_value):
        """Initialize a uniform distribution with minimum and maximum values.

        Args:
            min_value (float): The minimum value of the distribution.
            max_value (float): The maximum value of the distribution.

        """
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value

    def sample(self, shape: Iterable[int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Draws samples from a uniform distribution."""
        return torch.rand(*shape, device=device, dtype=dtype) * (self.max_value - self.min_value) + self.min_value


class RandDiscrete(BaseSigmaDistribution):
    """A discrete distribution for sampling noise levels in diffusion model training."""

    def __init__(self, values):
        """Initialize a discrete distribution with a list of possible values.

        Args:
            values (list or array-like): The possible values to sample from.

        """
        super().__init__()
        self.values = values

    def sample(self, shape: Iterable[int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Draws samples from a discrete distribution."""
        probs = [1 / len(self.values)] * len(self.values)  # set equal probability for all values
        return torch.tensor(np.random.choice(self.values, size=tuple(shape), p=probs), device=device, dtype=dtype)


class RandDiscreteUniform(BaseSigmaDistribution):
    """A discrete uniform distribution for sampling noise levels in diffusion model training."""

    def __init__(self, min_value, max_value):
        """Initialize a discrete uniform distribution with minimum and maximum values.

        Args:
            min_value (int): The minimum value (inclusive) of the distribution.
            max_value (int): The maximum value (exclusive) of the distribution.

        """
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value

    def sample(self, shape: Iterable[int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Draws samples from a discrete uniform distribution."""
        # torch.randint does not support dtype argument; output is always int64
        return torch.randint(low=self.min_value, high=self.max_value, size=tuple(shape), device=device).to(dtype)


class RandDiscreteLogUniform(BaseSigmaDistribution):
    """A discrete log-uniform distribution for sampling noise levels in diffusion model training."""

    def __init__(self, min_value, max_value, device="cpu", dtype=torch.float32):
        """Initialize the discrete log-uniform distribution with the given boundaries, device, and dtype."""
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value

    def sample(self, shape: Iterable[int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Draws samples from a discrete log-uniform distribution."""
        low = int(math.floor(math.log(self.min_value)))
        high = int(math.ceil(math.log(self.max_value)))
        return torch.randint(low=low, high=high, size=tuple(shape), device=device).to(dtype).exp()


class RandDiscreteLogLogistic(BaseSigmaDistribution):
    """A discrete log-logistic distribution for sampling noise levels in diffusion model training."""

    def __init__(self, loc=0.0, scale=1.0, min_value=0.0, max_value=float("inf")):
        """Initialize a discrete log-logistic distribution with location, scale, min, and max values.

        Args:
            loc (float): The location parameter of the distribution.
            scale (float): The scale parameter of the distribution.
            min_value (float): The minimum value for truncation.
            max_value (float): The maximum value for truncation.

        """
        super().__init__()
        self.loc = loc
        self.scale = scale
        self.min_value = min_value
        self.max_value = max_value

    def sample(self, shape: Iterable[int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Draws samples from an optionally truncated discrete log-logistic distribution."""
        min_value = torch.as_tensor(self.min_value, device=device, dtype=torch.float64)
        max_value = torch.as_tensor(self.max_value, device=device, dtype=torch.float64)
        min_cdf = min_value.log().sub(self.loc).div(self.scale).sigmoid()
        max_cdf = max_value.log().sub(self.loc).div(self.scale).sigmoid()
        u = torch.rand(*shape, device=device, dtype=torch.float64) * (max_cdf - min_cdf) + min_cdf
        return u.logit().mul(self.scale).add(self.loc).exp().to(dtype)


class RandDiscreteLogNormal(BaseSigmaDistribution):
    """A discrete log-normal distribution for sampling noise levels in diffusion model training."""

    def __init__(self, loc=0.0, scale=1.0):
        """Initialize a discrete log-normal distribution with given location and scale.

        Args:
            loc (float): The mean of the underlying normal distribution.
            scale (float): The standard deviation of the underlying normal distribution.

        """
        super().__init__()
        self.loc = loc
        self.scale = scale

    def sample(self, shape: Iterable[int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Draws samples from a discrete log-normal distribution."""
        return torch.log(torch.randn(*shape, device=device, dtype=dtype) * self.scale + self.loc)


class RandDiscreteNormal(BaseSigmaDistribution):
    """A discrete normal distribution for sampling noise levels in diffusion model training."""

    def __init__(self, loc=0.0, scale=1.0):
        """Initialize a discrete normal distribution with given location and scale.

        Args:
            loc (float): The mean of the normal distribution.
            scale (float): The standard deviation of the normal distribution.

        """
        super().__init__()


class RandDiscreteTruncatedNormal(BaseSigmaDistribution):
    """A truncated discrete normal distribution for sampling noise levels in diffusion model training."""

    def __init__(self, loc=0.0, scale=1.0, min_value=0.0, max_value=float("inf")):
        """Initialize a truncated discrete normal distribution with location, scale, min, and max values.

        Args:
            loc (float): The mean of the normal distribution.
            scale (float): The standard deviation of the normal distribution.
            min_value (float): The minimum value for truncation.
            max_value (float): The maximum value for truncation.

        """
        super().__init__()
        self.loc = loc
        self.scale = scale
        self.min_value = min_value
        self.max_value = max_value


class RandDiscreteTruncatedLogNormal(BaseSigmaDistribution):
    """A truncated discrete log-normal distribution for sampling noise levels in diffusion model training."""

    def __init__(self, loc=0.0, scale=1.0, min_value=0.0, max_value=float("inf")):
        """Initialize a truncated discrete log-normal distribution with location, scale, min, and max values.

        Args:
            loc (float): The mean of the underlying normal distribution.
            scale (float): The standard deviation of the underlying normal distribution.
            min_value (float): The minimum value for truncation.
            max_value (float): The maximum value for truncation.

        """
        super().__init__()
        self.loc = loc
        self.scale = scale
        self.min_value = min_value
        self.max_value = max_value

    def sample(self, shape: Iterable[int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Draws samples from a truncated discrete log-normal distribution."""
        min_value = torch.as_tensor(self.min_value, device=device, dtype=torch.float64)
        max_value = torch.as_tensor(self.max_value, device=device, dtype=torch.float64)
        min_cdf = (min_value.log() - self.loc) / self.scale
        max_cdf = (max_value.log() - self.loc) / self.scale
        u = torch.rand(*shape, device=device, dtype=torch.float64) * (max_cdf - min_cdf) + min_cdf
        return u.mul(self.scale).add(self.loc).exp().to(dtype)


class RandVDiffusion(BaseSigmaDistribution):
    """A distribution for sampling noise levels based on the v-diffusion training timestep distribution.

    Args:
        sigma_data (float): The scale parameter for the distribution.
        min_value (float): The minimum value for truncation.
        max_value (float): The maximum value for truncation.

    """

    def __init__(self, sigma_data=1.0, min_value=0.0, max_value=float("inf")):
        """Initialize the v-diffusion distribution with sigma_data, min_value, and max_value.

        Args:
            sigma_data (float): The scale parameter for the distribution.
            min_value (float): The minimum value for truncation.
            max_value (float): The maximum value for truncation.

        """
        super().__init__()
        self.sigma_data = sigma_data
        self.min_value = min_value
        self.max_value = max_value

    def sample(self, shape: Iterable[int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Draws samples from a truncated v-diffusion training timestep distribution."""
        min_cdf = math.atan(self.min_value / self.sigma_data) * 2 / math.pi
        max_cdf = math.atan(self.max_value / self.sigma_data) * 2 / math.pi
        u = torch.rand(*shape, device=device, dtype=dtype) * (max_cdf - min_cdf) + min_cdf
        return torch.tan(u * math.pi / 2) * self.sigma_data


class RandSplitLogNormal(BaseSigmaDistribution):
    """A split log-normal distribution for sampling noise levels in diffusion model training.

    This distribution samples from two log-normal distributions with different scales,
    choosing between them with equal probability.
    """

    def __init__(self, loc, scale_1, scale_2):
        """Initialize a split log-normal distribution with location and two scales.

        Args:
            loc (float): The mean of the underlying normal distribution.
            scale_1 (float): The standard deviation for the first half of the distribution.
            scale_2 (float): The standard deviation for the second half of the distribution.

        """
        super().__init__()
        self.loc = loc
        self.scale_1 = scale_1
        self.scale_2 = scale_2

    def sample(self, shape: Iterable[int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Draws samples from a split lognormal distribution."""
        n = torch.randn(*shape, device=device, dtype=dtype).abs()
        u = torch.rand(*shape, device=device, dtype=dtype)
        return torch.where(u < 0.5, n.mul(self.scale_1).add(self.loc).exp(), n.mul(self.scale_2).add(self.loc).exp())


class RandDiscreteCosineBeta(BaseSigmaDistribution):
    """A distribution based on the IDDPM beta schedule for sampling noise levels in diffusion model training."""

    def __init__(self, M: int = 20, C_1: float = 0.001, C_2: float = 0.008):
        """Initialize the IDDPM alpha distribution.

        Args:
            M (int): The number of discrete timesteps. Defaults to 20.
            C_1 (float): The C_1 parameter from the iDDPM cosine schedule, used for clipping. Defaults to 0.001.
            C_2 (float): The C_2 parameter from the iDDPM cosine schedule, used for scaling the freq of the cosine.
                Defaults to 0.008.

        """
        super().__init__()
        self.u = torch.zeros(M + 1)

        def alpha_bar(j):
            return (0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2

        for j in torch.arange(M, 0, -1):  # M, ..., 1
            self.u[j - 1] = ((self.u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1).sqrt()
        self.u = self.u[:-1]  # during training, we don't need to sample zero

    def sample(self, shape: Iterable[int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Draws samples from a discrete IDDPM alpha distribution."""
        if not self.u.dtype == dtype or not self.u.device == device:
            self.u = self.u.to(device=device)
            self.u = self.u.to(dtype=dtype)
        idx = torch.randint(0, len(self.u), size=tuple(shape), device=device)
        return self.u[idx]


class RandDiscreteLinearBeta(BaseSigmaDistribution):
    """Distribution based on classic DDPM linear beta schedule for sampling noise levels in diffusion model training."""

    def __init__(self, M: int = 20, beta_start: float = 0.0001, beta_end: float = 0.02):
        """Intialize the Linear discrete distribution defined in alpha/beta space.

        Args:
        M (int): Number of discrete timesteps (sigma levels). Defaults to 20.
        beta_start (float): Starting beta value. Defaults to 0.0001.
        beta_end (float): Final beta value. Defaults to 0.02.

        """
        super().__init__()
        betas = torch.linspace(beta_start, beta_end, M)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.sigmas = torch.sqrt((1.0 - alphas_cumprod) / alphas_cumprod)  # shape [M]

    def sample(self, shape: Iterable[int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Draws samples from the discrete DDPM linear beta sigma distribution."""
        sigmas = self.sigmas.to(device=device, dtype=dtype)
        idx = torch.randint(0, len(sigmas), size=tuple(shape), device=device)
        return sigmas[idx]


class DVLinearDistribution(BaseSigmaDistribution):
    """Noise distribution sampling t Uniform(min_t, max_t) and transforming to sigma via the DV "linear" schedule."""

    def __init__(
        self,
        beta0: float = 0.1,
        beta1: float = 20.0,
        sample_step_schedule: tuple = (0.001, 1),
        sigma_data: float = 1.0,
    ):
        """Initialize the uniform t noise scheduler with beta schedule parameters.

        Args:
        beta0 (float): initial beta at t = min_t. Defaults to 0.1.
        beta1 (float): final beta at t = max_t. Defaults to 20.0.
        sample_step_schedule (tuple): (min_t, max_t) for uniform sampling of t. Defaults to (0.001, 1).
        sigma_data (float): data scaling for EDM compatibility. Defaults to 1.0.

        """
        super().__init__()
        self.beta0 = beta0
        self.beta1 = beta1
        self.min_t = sample_step_schedule[0]
        self.max_t = sample_step_schedule[1]
        self.sigma_data = sigma_data

        # Compute sigma_max at t = min_t for initial noise scaling
        log_alpha_min = -(self.beta1 - self.beta0) / 4.0 * (self.min_t**2) - self.beta0 / 2.0 * self.min_t
        alpha_min = math.exp(log_alpha_min)
        sigma_min = math.sqrt(1.0 - alpha_min**2)
        snr_min = alpha_min**2 / sigma_min**2
        self.sigma_max = self.sigma_data / math.sqrt(snr_min)

    def sample(self, shape: Iterable[int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Draw sigma samples.

        The sampling procedure follows these steps:
        1. Sampling t = Uniform(min_t, max_t)
        2. Computing alpha(t) and sigma(t) via the quadratic β schedule
        3. Converting to EDM sigma via SNR matching
        """
        # 1. Uniform t
        u = torch.rand(*shape, device=device, dtype=dtype)
        t = u * (self.max_t - self.min_t) + self.min_t

        # 2. α(t) from quadratic β schedule
        log_alpha = -(self.beta1 - self.beta0) / 4.0 * (t**2) - self.beta0 / 2.0 * t
        alpha = log_alpha.exp()
        their_sigma = (1.0 - alpha**2).sqrt()

        # 3. Convert via SNR: edm_sigma = σ_data / sqrt(α² / their_sigma²)
        snr = alpha**2 / their_sigma**2
        sigmas = self.sigma_data / torch.sqrt(snr)

        return sigmas
