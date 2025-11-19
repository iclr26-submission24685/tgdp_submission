"""Noise schedulers for diffusion models."""

# Adapted from Scheikl et al. - Movement Primitive Diffusion
# https://github.com/ScheiklP/movement-primitive-diffusion/blob/main/movement_primitive_diffusion/noise_schedulers.py

import math
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch


def append_zero_to_tensor(action):
    """Append a zero to the end of the given tensor.

    Args:
        action (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The input tensor with a zero appended at the end.

    """
    return torch.cat([action, action.new_zeros([1])])


class BaseNoiseScheduler(ABC):
    """Base class for noise schedulers."""

    sigma_min: float
    sigma_max: float

    def __init__(self):
        """Initialize the BaseNoiseScheduler."""
        super().__init__()

    @abstractmethod
    def get_sigmas(
        self,
        n: int,
        append_zero: bool = True,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Return the noise schedule sigmas for n steps.

        Args:
            n (int): Number of steps.
            append_zero (bool): Whether to append a zero at the end of the sigmas. Defaults to True.
            device (torch.device): Device on which to create the sigmas. Defaults to torch.device("cpu").
            dtype (torch.dtype, optional): Data type for the sigmas. Defaults to torch.float32.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The noise schedule sigmas.

        """
        raise NotImplementedError


############### Schedulers defined in sigma space. ###############


class KarrasNoiseScheduler(BaseNoiseScheduler):
    """Noise scheduler implementing the Karras et al. (2022) schedule.

    This scheduler generates a noise schedule using a parameterized formula
    with parameters sigma_min, sigma_max, and rho, as described in Karras et al. (2022).
    """

    def __init__(self, sigma_min: float = 0.002, sigma_max: float = 80.0, rho=7.0):
        """Initialize the KarrasNoiseScheduler.

        Args:
            sigma_min (float): The minimum noise level. Defaults to 0.002.
            sigma_max (float): The maximum noise level. Defaults to 80.0.
            rho (float): The parameter controlling the noise schedule shape. Defaults to 7.0.

        """
        super().__init__()
        # "setting rho = 3 nearly equalizes the truncation error at each step, but that rho in range of 5 to 10 performs
        # much better for sampling images. This suggests that errors near rho_min have a large impact.
        # We set rho = 7 for the remainder of this paper" - Karras et al.
        self.rho = rho
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def get_sigmas(
        self,
        n,
        append_zero: bool = True,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Construct the noise schedule of Karras et al. (2022).

        Args:
            n (int): Number of steps.
            append_zero (bool): Whether to append a zero at the end of the sigmas. Defaults to True.
            device (torch.device): Device on which to create the sigmas. Defaults to torch.device("cpu").
            dtype (torch.dtype): Data type for the sigmas. Defaults to torch.float32.

        Returns:
            torch.Tensor: The noise schedule sigmas.

        """
        ramp = torch.linspace(0, 1, n, dtype=dtype, device=device)
        min_inv_rho = self.sigma_min ** (1 / self.rho)
        max_inv_rho = self.sigma_max ** (1 / self.rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.rho
        if append_zero:
            return append_zero_to_tensor(sigmas)
        else:
            return sigmas


class ExponentialNoiseScheduler(BaseNoiseScheduler):
    """Noise scheduler implementing an exponential decay schedule between sigma_max and sigma_min."""

    def __init__(
        self,
        sigma_min: float,
        sigma_max: float,
    ):
        """Initialize the ExponentialNoiseScheduler.

        Args:
            sigma_min (float): The minimum noise level.
            sigma_max (float): The maximum noise level.

        """
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def get_sigmas(
        self,
        n: int,
        append_zero: bool = True,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Return the exponential noise schedule sigmas for n steps.

        Args:
            n (int): Number of steps.
            append_zero (bool): Whether to append a zero at the end of the sigmas. Defaults to True.
            device (torch.device): Device on which to create the sigmas. Defaults to torch.device("cpu").
            dtype (torch.dtype): Data type for the sigmas. Defaults to torch.float32.

        Returns:
            torch.Tensor: The noise schedule sigmas.

        """
        sigmas = torch.linspace(
            math.log(self.sigma_max),
            math.log(self.sigma_min),
            n,
            dtype=dtype,
            device=device,
        ).exp()
        if append_zero:
            return append_zero_to_tensor(sigmas)
        else:
            return sigmas


class LinearNoiseScheduler(BaseNoiseScheduler):
    """Noise scheduler implementing a linear schedule between sigma_max and sigma_min."""

    def __init__(self, sigma_min: float, sigma_max: float):
        """Initialize the LinearNoiseScheduler.

        Args:
            sigma_min (float): The minimum noise level.
            sigma_max (float): The maximum noise level.

        """
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def get_sigmas(
        self,
        n: int,
        append_zero: bool = True,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Return the linear noise schedule sigmas for n steps.

        Args:
            n (int): Number of steps.
            append_zero (bool): Whether to append a zero at the end of the sigmas. Defaults to True.
            device (torch.device): Device on which to create the sigmas. Defaults to torch.device("cpu").
            dtype (torch.dtype): Data type for the sigmas. Defaults to torch.float32.

        Returns:
            torch.Tensor: The noise schedule sigmas.

        """
        sigmas = torch.linspace(self.sigma_max, self.sigma_min, n, dtype=dtype, device=device)
        if append_zero:
            return append_zero_to_tensor(sigmas)
        else:
            return sigmas


class PolyNoiseScheduler(BaseNoiseScheduler):
    """Noise scheduler implementing a polynomial schedule in log sigma space."""

    def __init__(self, sigma_min: float, sigma_max: float, rho=1.0):
        """Initialize the PolyNoiseScheduler.

        Args:
            sigma_min (float): The minimum noise level.
            sigma_max (float): The maximum noise level.
            rho (float): The exponent for the polynomial schedule. Defaults to 1.0.

        """
        super().__init__()
        self.rho = rho
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def get_sigmas(
        self, n: int, append_zero: bool = True, device=torch.device("cpu"), dtype=torch.float32
    ) -> torch.Tensor:
        """Return a polynomial in log sigma noise schedule."""
        ramp = torch.linspace(1, 0, n, dtype=dtype, device=device) ** self.rho
        sigmas = torch.exp(ramp * (math.log(self.sigma_max) - math.log(self.sigma_min)) + math.log(self.sigma_min))

        if append_zero:
            return append_zero_to_tensor(sigmas)
        else:
            return sigmas


class DVLinearScheduler(BaseNoiseScheduler):
    """Noise scheduler implementing the "linear" schedule used in Decision Veteran (Lu et al., 2025)."""

    def __init__(
        self, beta0: float = 0.1, beta1: float = 20.0, sample_step_schedule: tuple = (0.001, 1), sigma_data: float = 1.0
    ):
        """Initialize the AlphaBetaNoiseScheduler.

        Args:
            beta0 (float): The initial beta value. Defaults to 0.1.
            beta1 (float): The final beta value. Defaults to 20.0.
            sample_step_schedule (tuple): The schedule for sampling steps. Defaults to (0.001, 1).
            sigma_data (float): The data scaling parameter for EDM compatibility. Defaults to 1.0.

        """
        super().__init__()
        self.beta0 = beta0
        self.beta1 = beta1
        self.sample_step_schedule = sample_step_schedule
        self.sigma_data = sigma_data

        # Compute sigma_max (at t=1.0) for initial sample creation.
        log_alpha_max = -(beta1 - beta0) / 4.0 * (sample_step_schedule[1] ** 2) - beta0 / 2.0 * sample_step_schedule[1]
        alpha_max = math.exp(log_alpha_max)
        sigma_max = math.sqrt(1.0 - alpha_max**2)
        snr_max = alpha_max**2 / sigma_max**2
        self.sigma_max = sigma_data / math.sqrt(snr_max)

    def get_sigmas(
        self, n: int, append_zero: bool = True, device=torch.device("cpu"), dtype=torch.float32
    ) -> torch.Tensor:
        """Return noise schedule that exactly matches the alpha/beta formulation."""
        # Generate time points. Lu et al. do not set the final sigma to zero.
        n = n + 1 if not append_zero else n
        t_diffusion = torch.linspace(
            self.sample_step_schedule[1], self.sample_step_schedule[0], n + 1, dtype=dtype, device=device
        )

        # Compute log_alpha using their quadratic beta ("linear") schedule.
        log_alpha = -(self.beta1 - self.beta0) / 4.0 * (t_diffusion**2) - self.beta0 / 2.0 * t_diffusion
        alpha = log_alpha.exp()

        # Compute their sigma schedule and convert to EDM-compatible sigmas.
        their_sigma = (1.0 - alpha**2).sqrt()
        snr = alpha**2 / their_sigma**2
        sigmas = self.sigma_data / torch.sqrt(snr)

        if append_zero:
            return append_zero_to_tensor(sigmas)
        else:
            return sigmas


class VENoiseScheduler(BaseNoiseScheduler):
    """Noise scheduler implementing the Variance Exploding (VE) schedule.

    This scheduler generates a continuous noise schedule where the noise variance increases
    exponentially from sigma_min to sigma_max, as used in VE SDEs.
    """

    def __init__(self, sigma_min: float, sigma_max: float):
        """Initialize the VENoiseScheduler.

        Args:
            sigma_min (float): The minimum noise level.
            sigma_max (float): The maximum noise level.

        """
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def get_sigmas(
        self,
        n: int,
        append_zero: bool = True,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Return the VE noise schedule sigmas for n steps.

        The schedule increases the noise variance exponentially from sigma_min to sigma_max.

        Args:
            n (int): Number of steps.
            append_zero (bool): Whether to append a zero at the end of the sigmas. Defaults to True.
            device (torch.device): Device on which to create the sigmas. Defaults to torch.device("cpu").
            dtype (torch.dtype): Data type for the sigmas. Defaults to torch.float32.

        Returns:
            torch.Tensor: The noise schedule sigmas.

        """
        t = torch.linspace(1, 0, n, dtype=dtype, device=device)
        sigmas = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        if append_zero:
            return append_zero_to_tensor(sigmas)
        else:
            return sigmas


####### Schedulers defined in alpha/beta spece. The are used for compatibility with DDPM/DDIM frameworks. #######


class CosineBetaNoiseScheduler(BaseNoiseScheduler):
    """Noise scheduler implementing the IDDPM cosine beta schedule in the Karras/EDM noise domain.

    This implements the IDDPM noise scheduler originally proposed in Nichol and Dhariwal (2021).
    """

    def __init__(
        self,
        sigma_min: Optional[float] = None,
        sigma_max: Optional[float] = None,
        M: int = 20,
        j_0: int = 0,
        C_1: float = 0.001,
        C_2: float = 0.008,
    ):
        """Initialize the IDDPM noise scheduler originally proposed in Nichol and Dhariwal (2021).

        This formulation is based on the derivations from Karras et al. (2022) [Appendix C3.2].

        Args:
            sigma_min (float): The minimum noise level. If None, we use the smallest value larger than 0 from the IDDPM
                schedule.
            sigma_max (float): The maximum noise level. If None, we use the largest value from the IDDPM schedule, given
                the value for j_0.
            M (int): The number of steps in the IDDPM noise schedule at training time.
            j_0 (int): The starting index of the IDDPM noise schedule. The original schedule starts out with very high
                values. From j_0=8, we are approx in the range of the Karras noise schedule (sigma<=80)
            C_1 (float): The C_1 parameter from the iDDPM cosine schedule, used for clipping. Defaults to 0.001.
            C_2 (float): The C_2 parameter from the iDDPM cosine schedule, used for scaling the freq of the cosine.
                Defaults to 0.008.

        """
        super().__init__()
        self.M = M
        self.j_0 = j_0
        self.C_1 = C_1
        self.C_2 = C_2

        # Compute the IDDPM noise schedule.
        u = torch.zeros(self.M + 1)

        def alpha_bar(j):
            return (0.5 * np.pi * j / self.M / (self.C_2 + 1)).sin() ** 2

        for j in torch.arange(self.M, self.j_0, -1):  # M, ..., 1
            u[j - 1] = ((u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=self.C_1) - 1).sqrt()
        # We remove the zero value at the end and put it back in when append_zero=True. If append_zero=False, we get n
        # noise levels, but since the last is the final one, the sampler will only do n-1 evaluations. Therefore we
        # should usually append_zero for sampling. This is in line with the API of other schedulers.
        u = u[:-1]
        if sigma_min is None:
            sigma_min = float(u[-1])
        if sigma_max is None or sigma_max > u[self.j_0]:
            sigma_max = float(u[self.j_0])
        self.u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
        self.sigma_min = self.u_filtered.min().item()
        self.sigma_max = self.u_filtered.max().item()

    def get_sigmas(
        self, n: int, append_zero: bool = True, device=torch.device("cpu"), dtype=torch.float32
    ) -> torch.Tensor:
        """Construct a continuous IDDPM noise schedule.

        Args:
            n (int): Number of steps.
            append_zero (bool): Whether to append a zero at the end of the sigmas. Defaults to True.
            device (torch.device): Device on which to create the sigmas. Defaults to torch.device("cpu").
            dtype (torch.dtype): Data type for the sigmas. Defaults to torch.float32.

        Returns:
            torch.Tensor: The noise schedule sigmas.

        """
        if self.u_filtered.device != device or self.u_filtered.dtype != dtype:
            self.u_filtered = self.u_filtered.to(device=device, dtype=dtype)
        # (sigma_max ** 2) * ((sigma_min ** 2 / sigma_max ** 2) ** (step_indices / (num_steps - 1)))
        step_indices = torch.arange(n, device=device)
        sigmas = self.u_filtered[((len(self.u_filtered) - 1) / (n - 1) * step_indices).round().to(torch.int64)]
        if append_zero:
            sigmas = append_zero_to_tensor(sigmas)
        return sigmas


class LinearBetaNoiseScheduler(BaseNoiseScheduler):
    """Noise scheduler implementing the classic DDPM linear beta schedule in the Karras/EDM noise domain."""

    def __init__(
        self,
        beta_start: float = 0.0001,
        beta_end: float = 0.6,
        M: int = 20,
        sigma_min: Optional[float] = None,
        sigma_max: Optional[float] = None,
    ):
        """Initialize the IDDMPLinearNoiseScheduler.

        Args:
            beta_start (float): Start value for linear betas (default 0.0001).
            beta_end (float): End value for linear betas (default 0.6 - This is higher than usual since we use fewer
                diffusion steps: 20 instead of 1000).
            M (int): Number of diffusion timesteps.
            sigma_min (Optional[float]): Minimum sigma. Clips values below this.
            sigma_max (Optional[float]): Maximum sigma. Clips values above this.

        """
        super().__init__()
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.M = M

        betas = torch.linspace(beta_start, beta_end, M)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        sigmas = torch.sqrt((1.0 - alphas_cumprod) / alphas_cumprod)
        sigmas = torch.flip(sigmas, dims=[0])

        # Clip to range if needed
        sigmas_filtered = sigmas
        if sigma_min is not None:
            sigmas_filtered = sigmas_filtered[sigmas_filtered >= sigma_min]
        if sigma_max is not None:
            sigmas_filtered = sigmas_filtered[sigmas_filtered <= sigma_max]
        self.sigmas_filtered = sigmas_filtered
        self.sigma_min = sigmas_filtered.min().item()
        self.sigma_max = sigmas_filtered.max().item()

    def get_sigmas(
        self, n: int, append_zero: bool = True, device: torch.device = torch.device("cpu"), dtype=torch.float32
    ) -> torch.Tensor:
        """Return the linear beta noise schedule sigmas for n steps.

        Args:
            n (int): Number of steps.
            append_zero (bool): Whether to append a zero at the end of the sigmas. Defaults to True.
            device (torch.device): Device on which to create the sigmas. Defaults to torch.device("cpu").
            dtype (torch.dtype): Data type for the sigmas. Defaults to torch.float32.

        Returns:
            torch.Tensor: The noise schedule sigmas.

        """
        if self.sigmas_filtered.device != device or self.sigmas_filtered.dtype != dtype:
            self.sigmas_filtered = self.sigmas_filtered.to(device=device, dtype=dtype)
        choices = torch.linspace(0, len(self.sigmas_filtered) - 1, n).round().to(torch.int64)
        sigmas = self.sigmas_filtered[choices]
        if append_zero:
            sigmas = torch.cat([sigmas, torch.zeros(1, device=device, dtype=dtype)])
        return sigmas.to(device=device, dtype=dtype)
