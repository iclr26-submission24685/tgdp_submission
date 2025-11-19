"""Implements various scaling (also called preconditioning) utilities for diffusion models as described in Karras et al.

Adapted from Scheikl et al. - Movement Primitive Diffusion
https://github.com/ScheiklP/movement-primitive-diffusion/blob/main/movement_primitive_diffusion/models/scaling.py

"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch


class Scaling(ABC, torch.nn.Module):
    """Abstract base class for scaling (preconditioning) utilities in diffusion models.

    Subclasses must implement the __call__ method to return scaling factors for the diffusion model.
    """

    @abstractmethod
    def __call__(self, sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return c_skip, c_out, c_in, c_noise scaling factors for the diffusion model.

        Args:
            sigma (torch.Tensor): A tensor representing the noise levels.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - c_skip (torch.Tensor): The scaling factor for the skip connection.
                - c_out (torch.Tensor): The scaling factor for the output.
                - c_in (torch.Tensor): The scaling factor for the input.
                - c_noise (torch.Tensor): The scaling factor for the noise.

        """
        raise NotImplementedError


class KarrasScaling(Scaling):
    """Implements the scaling (preconditioning) utility as described in Karras et al. for diffusion models.

    This scaling is used to condition the model on the noise level (sigma) and the data standard deviation (sigma_data).
    This lets the model predict something in between the absolute and and epsilon, depending on the sigma value.

    """

    def __init__(self, sigma_data: Optional[float] = 1.0) -> None:
        """Initialize the scaling utility with the given sigma_data value.

        Args:
            sigma_data (Optional[float]): The data standard deviation used for scaling. Defaults to 1.0 since we assume
                Gaussian normalization.

        """
        super().__init__()
        self.sigma_data = sigma_data

    def set_sigma_data(self, sigma_data: float) -> None:
        """Set the sigma_data attribute.

        Args:
            sigma_data (float): The data standard deviation used for scaling.

        """
        self.sigma_data = sigma_data

    def __call__(self, sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute scaling coefficients for Karras scaling.

        Args:
            sigma (torch.Tensor): A tensor representing the noise levels.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - c_skip: Skip connection scaling factor.
                - c_out: Output scaling factor.
                - c_in: Input scaling factor.
                - c_noise: Noise input for the model.

        """
        if self.sigma_data is None:
            raise RuntimeError("Sigma data must be set before calling KarrasScaling")
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1.0 / (sigma**2 + self.sigma_data**2) ** 0.5
        eps = 1e-9 * (sigma == 0.0)
        c_noise = torch.log(sigma + eps) / 4.0  # avoid log(0)

        return c_skip, c_out, c_in, c_noise


class TrajectoryKarrasScalingAbsolute(KarrasScaling):
    """Implements the absolute variant of Karras scaling for trajectory diffusion models.

    This scaling configures the model to predict the absolute value of the denoised sample but computes the noise
    level input in the same way as the original Karras scaling.
    """

    def __call__(self, sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute scaling coefficients for the absolute variant of Karras scaling.

        Args:
            sigma (torch.Tensor): A tensor representing the noise levels.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - c_skip: Skip connection scaling factor (zeros).
                - c_out: Output scaling factor (ones).
                - c_in: Input scaling factor.
                - c_noise: Noise input for the model.

        """
        if self.sigma_data is None:
            raise RuntimeError("Sigma data must be set before calling KarrasScaling")
        c_skip = torch.zeros_like(sigma)
        c_out = torch.ones_like(sigma)
        c_in = 1.0 / (sigma**2 + self.sigma_data**2) ** 0.5
        eps = 1e-9 * (sigma == 0.0)
        c_noise = torch.log(sigma + eps) / 4.0  # avoid log(0)

        return c_skip, c_out, c_in, c_noise


class TrajectoryKarrasScalingEpsilon(KarrasScaling):
    """Implements the epsilon variant of Karras scaling for trajectory diffusion models.

    This scaling configures the model to predict the noise (epsilon) for the denoised sample,
    while computing the noise level input as in the original Karras scaling.
    """

    def __call__(self, sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute scaling coefficients for the epsilon variant of Karras scaling.

        Args:
            sigma (torch.Tensor): A tensor representing the noise levels.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - c_skip: Skip connection scaling factor (ones).
                - c_out: Output scaling factor (sigma).
                - c_in: Input scaling factor.
                - c_noise: Noise input for the model.

        """
        if self.sigma_data is None:
            raise RuntimeError("Sigma data must be set before calling KarrasScaling")
        c_skip = torch.ones_like(sigma)
        c_out = sigma
        c_in = 1.0 / (sigma**2 + self.sigma_data**2) ** 0.5
        eps = 1e-9 * (sigma == 0.0)
        c_noise = torch.log(sigma + eps) / 4.0  # avoid log(0)

        return c_skip, c_out, c_in, c_noise


class IDDPMScaling(Scaling, ABC):
    """Implements the scaling (preconditioning) utility for IDDPM (Improved Denoising Diffusion Probabilistic Models).

    This class computes the scaling coefficients and noise schedule as described in Janner et al..
    """

    def __init__(
        self,
        sigma_min: Optional[float] = None,
        sigma_max: Optional[float] = None,
        M: int = 20,
        j_0: int = 0,
        C_1: float = 0.001,
        C_2: float = 0.008,
    ) -> None:
        """Initialize the IDDPM scaling utility.

        Args:
            sigma_min (Optional[float]): The minimum noise level. Defaults to None.
            sigma_max (Optional[float]): The maximum noise level. Defaults to None.
            M (int): The number of noise levels used for training. Defaults to 20.
            j_0 (int): The index of the noise level to start from for inference. Defaults to 0.
            C_1 (float): The C_1 parameter from the iDDPM cosine schedule, used for clipping. Defaults to 0.001.
            C_2 (float): The C_2 parameter from the iDDPM cosine schedule, used for scaling the freq of the cosine.
                Defaults to 0.008.

        """
        super().__init__()
        step_indices = torch.arange(M)
        self.u = torch.zeros(M + 1)

        def alpha_bar(j):
            return (0.5 * torch.pi * j / M / (C_2 + 1)).sin() ** 2

        for j in torch.arange(M, j_0, -1):  # M, ..., 1, u(M) = 0
            self.u[j - 1] = ((self.u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1).sqrt()
        self.u = self.u[
            :-1
        ]  # In DDPM, t is the index of the noise level we are stepping to (t=0 correstponds to sigma_min!=0)
        if sigma_min is None:
            sigma_min = float(self.u[-1])
        if sigma_max is None:
            sigma_max = float(self.u[j_0])
        self.u = self.u[torch.logical_and(self.u >= sigma_min, self.u <= sigma_max)]
        self.u = self.u[((len(self.u) - 1) / (M - 1) * step_indices).round().to(torch.int64)]
        self.u = self.u.flip(0)  # In DDPM et al we go from M-1 to 0


class IDDPMEpsilonScaling(IDDPMScaling):
    """Implements the epsilon variant of IDDPM scaling, where the neural network predicts the noise component."""

    def __call__(self, sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute scaling coefficients for IDDPM scaling, in that the NN predicts the noise.

        Args:
            sigma (torch.Tensor): A tensor representing the noise levels.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - c_skip: Skip connection.
                - c_out: Output scaling.
                - c_in: Input scaling.
                - c_noise: Noise input which is used as an input to the model.

        """
        # put u on the same device as sigma
        if not self.u.device == sigma.device:
            self.u = self.u.to(sigma.device)

        # compute scaling coefficients
        c_skip = torch.ones_like(sigma)
        c_out = sigma
        c_in = 1.0 / (sigma**2 + 1) ** 0.5
        c_noise = (
            torch.argmin((sigma.unsqueeze(-1) - self.u.unsqueeze(0)).abs(), dim=-1).squeeze(0).float()
        )  # we condition with t=M,...,1

        return c_skip, c_out, c_in, c_noise


class IDDPMAbsoluteScaling(IDDPMScaling):
    """Implements the absolute variant of IDDPM scaling, where the neural network predicts the absolute output."""

    def __call__(self, sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute scaling coefficients for IDDPM scaling, in that the NN predicts the absolute output.

        Args:
            sigma (torch.Tensor): A tensor representing the noise levels.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - c_skip: Skip connection.
                - c_out: Output scaling.
                - c_in: Input scaling.
                - c_noise: Noise input which is used as an input to the model.

        """
        # put u on the same device as sigma
        if not self.u.device == sigma.device:
            self.u = self.u.to(sigma.device)

        # compute scaling coefficients
        c_skip = torch.zeros_like(sigma)
        c_out = torch.ones_like(sigma)
        c_in = 1.0 / (sigma**2 + 1) ** 0.5
        c_noise = (
            torch.argmin((sigma.unsqueeze(-1) - self.u.unsqueeze(0)).abs(), dim=-1).squeeze(0).float()
        )  # we condition with t = M,...,1

        return c_skip, c_out, c_in, c_noise


class DDPMScaling(Scaling, ABC):
    """Implements the scaling (preconditioning) utility for linear DDPM schedules."""

    def __init__(
        self,
        beta_start: float = 0.0001,
        beta_end: float = 0.6,
        M: int = 20,
        sigma_min: Optional[float] = None,
        sigma_max: Optional[float] = None,
    ):
        """Initialize the DDPM scaling.

        Args:
            beta_start (float): Starting beta value for linear DDPM schedule.
            beta_end (float): Ending beta value for linear DDPM schedule.
            M (int): Number of noise levels (timesteps).
            sigma_min (Optional[float]): Minimum sigma to keep.
            sigma_max (Optional[float]): Maximum sigma to keep.

        """
        super().__init__()
        betas = torch.linspace(beta_start, beta_end, M)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        sigmas = torch.sqrt((1.0 - alphas_cumprod) / alphas_cumprod)

        # Optionally filter sigma range
        if sigma_min is not None:
            sigmas = sigmas[sigmas >= sigma_min]
        if sigma_max is not None:
            sigmas = sigmas[sigmas <= sigma_max]

        self.u = sigmas
        # Evenly space over [0, M-1]
        step_indices = torch.arange(len(self.u))
        self.u = self.u[((len(self.u) - 1) / max(len(self.u) - 1, 1) * step_indices).round().to(torch.int64)]
        self.u = self.u.flip(0)  # reverse order: high to low noise


class DDPMEpsilonScaling(DDPMScaling):
    """Scaling for DDPM epsilon (noise) prediction."""

    def __call__(self, sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute scaling coefficients for DDPM scaling, in that the NN predicts the epsilon output.

        Args:
            sigma (torch.Tensor): A tensor representing the noise levels.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - c_skip: Skip connection.
                - c_out: Output scaling.
                - c_in: Input scaling.
                - c_noise: Noise input which is used as an input to the model.

        """
        if self.u.device != sigma.device:
            self.u = self.u.to(sigma.device)
        c_skip = torch.ones_like(sigma)
        c_out = sigma
        c_in = 1.0 / (sigma**2 + 1).sqrt()
        c_noise = torch.argmin((sigma.unsqueeze(-1) - self.u.unsqueeze(0)).abs(), dim=-1).squeeze(0).float()
        return c_skip, c_out, c_in, c_noise


class DDPMAbsoluteScaling(DDPMScaling):
    """Scaling for DDPM absolute (x0) prediction."""

    def __call__(self, sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute scaling coefficients for DDPM scaling, in that the NN predicts the absolute output.

        Args:
            sigma (torch.Tensor): A tensor representing the noise levels.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - c_skip: Skip connection.
                - c_out: Output scaling.
                - c_in: Input scaling.
                - c_noise: Noise input which is used as an input to the model.

        """
        if self.u.device != sigma.device:
            self.u = self.u.to(sigma.device)
        c_skip = torch.zeros_like(sigma)
        c_out = torch.ones_like(sigma)
        c_in = 1.0 / (sigma**2 + 1).sqrt()
        c_noise = torch.argmin((sigma.unsqueeze(-1) - self.u.unsqueeze(0)).abs(), dim=-1).squeeze(0).float()
        return c_skip, c_out, c_in, c_noise


class AbsoluteScaling(Scaling):
    """Implements the Absolute scaling, where the neural network predicts the absolute value of the denoised sample."""

    def __call__(self, sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute coefficients for Absolute scaling, in that the NN predicts the absolute value of the denoised sample.

        Args:
            sigma (torch.Tensor): A tensor representing the noise levels.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - c_skip: Skip connection.
                - c_out: Output scaling.
                - c_in: Input scaling.
                - c_noise: Noise input which is used as an input to the model.

        """
        c_skip = torch.zeros_like(sigma)
        c_out = torch.ones_like(sigma)
        c_in = torch.ones_like(sigma)
        c_noise = sigma

        return c_skip, c_out, c_in, c_noise


class EpsilonScaling(Scaling):
    """Implements the Epsilon scaling, where the neural network predicts the epsilon (noise) of the denoised sample."""

    def __call__(self, sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute coefficients for Epsilon scaling, in that the NN predicts the epsilon of the denoised sample.

        Args:
            sigma (torch.Tensor): A tensor representing the noise levels.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - c_skip: Skip connection.
                - c_out: Output scaling.
                - c_in: Input scaling.
                - c_noise: Noise input which is used as an input to the model.

        """
        c_skip = torch.ones_like(sigma)
        c_out = torch.ones_like(sigma)
        c_in = torch.ones_like(sigma)
        c_noise = sigma

        return c_skip, c_out, c_in, c_noise
