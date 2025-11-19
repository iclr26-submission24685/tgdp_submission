"""Euler Sampler."""

# Adapted from Scheikl et. al. - Movement Primitive Diffusion
# https://github.com/ScheiklP/movement-primitive-diffusion/blob/main/movement_primitive_diffusion/samplers.euler.py

import logging
from typing import Optional, Tuple

import torch
from tensordict import TensorDict

from ..inpainting_conditioning import ConditionType
from ..models.diffusion.base_diffusion import BaseDiffusionModel
from .base_sampler import BaseSampler

logger = logging.getLogger(__name__)


class EulerSampler(BaseSampler):
    """Euler sampler.

    Implements a variant of Algorithm 2 (Euler steps) from Karras et al. (2022).
    Stochastic sampler, which combines a first order ODE solver with explicit Langevin-like "churn"
    of adding and removing noise.
    Every update consists of these substeps:
    1. Addition of noise given the factor eps
    2. Solving the ODE dx/dt at timestep t using the score model
    3. Take Euler step from t -> t+1 to get x_{i+1}

    In contrast to the Heun sampler, this sampler does not use a correction step.
    """

    def __init__(self, s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0):
        """Initialize the EulerSampler.

        Args:
            s_churn (float): Amount of stochasticity to add during sampling.
            s_tmin (float): Minimum sigma value for churn.
            s_tmax (float): Maximum sigma value for churn.
            s_noise (float): Multiplier for the noise added to the sample. This does not affect the noise level provided
                to the model, but rather the noise added to the sample before the Euler step.

        """
        super().__init__()

        self.s_churn = s_churn
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise

    @torch.no_grad()
    def sample(
        self,
        model: BaseDiffusionModel,
        noisy_sample: torch.Tensor,
        sigmas: torch.Tensor,
        conditions: Optional[ConditionType] = None,
        extra_inputs: Optional[TensorDict] = None,
        return_steps: bool = False,
    ) -> Tuple[torch.Tensor, TensorDict]:
        """Sample loop to calculate the noise free sample.

        Args:
            model: Diffusion model to be used for sampling.
            noisy_sample: Initial noisy sample. [Batch, Time, Features]
            sigmas: Sigma values to iterate over during sampling. [Steps]
            conditions: Dictionary containing the conditions.
            extra_inputs: Extra inputs to the model.
            return_steps: If True, return the intermediate steps of the diffusion process. This includes gradients and
                intermediate samples and values.

        Returns:
            Denoised sample. [Batch, Time, Features]
            infos: Dictionary containing additional information. If return_steps is False, this is an empty TensorDict.

        """
        # Initialize dictionary for steps info.
        if return_steps:
            infos = TensorDict({"steps": torch.empty((len(sigmas), *noisy_sample.shape), device=noisy_sample.device)})
            infos["steps"][0] = noisy_sample
        else:
            infos = TensorDict({})

        # Iterate over all sigma values and apply the Euler step.
        for i in range(len(sigmas) - 1):
            gamma = (
                min(self.s_churn / (len(sigmas) - 1), 2**0.5 - 1) if self.s_tmin <= sigmas[i] <= self.s_tmax else 0.0
            )
            eps = torch.randn_like(noisy_sample) * self.s_noise
            sigma_hat = sigmas[i] * (gamma + 1)  # Add noise to sigma.

            if gamma > 0:  # If gamma > 0, use additional noise level for computation.
                noisy_sample = noisy_sample + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
            denoised, step_info = model(
                noisy_sample, sigma_hat, conditions, extra_inputs, diffusion_step=i, return_info=return_steps
            )
            assert sigma_hat.ndim == 0, "Sigma_hat should be a scalar."
            d = (noisy_sample - denoised) / sigma_hat
            dt = sigmas[i + 1] - sigma_hat
            noisy_sample = noisy_sample + d * dt

            # Append step info to info dictionary. All values in the info TensorDict are also TensorDicts, except steps.
            if return_steps:
                for k1, v1 in step_info.items():
                    if i == 0:  # Initialize the info dictionary.
                        infos[k1] = TensorDict(
                            {k2: torch.empty((len(sigmas) - 1, *v2.shape), device=v2.device) for k2, v2 in v1.items()}
                        )
                    for k2, v2 in v1.items():
                        infos[k1][k2][i] = v2
                infos["steps"][i + 1] = noisy_sample

        # Finally, we obtain our denoised sample.
        denoised_sample = noisy_sample

        return denoised_sample, infos.detach()
