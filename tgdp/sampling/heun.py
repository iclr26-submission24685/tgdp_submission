"""Heun Sampler."""

# Adapted from Scheikl et. al. - Movement Primitive Diffusion
# https://github.com/ScheiklP/movement-primitive-diffusion/blob/main/movement_primitive_diffusion/samplers.heun.py

import logging
from typing import Optional, Tuple

import torch
from tensordict import TensorDict

from ..inpainting_conditioning import ConditionType
from ..models.diffusion.base_diffusion import BaseDiffusionModel
from .base_sampler import BaseSampler

logger = logging.getLogger(__name__)


class HeunSampler(BaseSampler):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022).

    For S_churn = 0 this is an ODE solver otherwise SDE
    Every update consists of these substeps:
    1. Addition of noise given the factor eps
    2. Solving the ODE dx/dt at timestep t using the score model
    3. Take Euler step from t -> t+1 to get x_{i+1}
    4. 2nd order correction step to get x_{i+1}^{(2)}

    In contrast to the Euler variant, this variant computes a 2nd order correction step.
    """

    def __init__(self, s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0):
        """Initialize the Heun sampler with the given parameters.

        Args:
            s_churn (float, optional): Langevin-like churn of adding and removing noise (Karras et al.). Default is 0.0.
            s_tmin (float, optional): Minimum noise level for that stochasticity is applied. Default is 0.0.
            s_tmax (float, optional): Maximum noise level for that stochasticity is applied. Default is inf.
            s_noise (float, optional): Noise std multiplier. Setting this slightly above 1 can counteract the tendency
                of diffusion models to remove too much noise and therefore increase fidelity. Default is 1.0.

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
            logger.warning(
                "Return steps for the Heun sample is not properly implemented. It will return the targets and gradients"
                " of the second evaluation step only."
            )
            # TODO Implement return_steps for Heun sampler
            infos = TensorDict({"steps": torch.empty((len(sigmas), *noisy_sample.shape), device=noisy_sample.device)})
            infos["steps"][0] = noisy_sample
        else:
            infos = TensorDict({})

        # Initialize dictionary for steps info.
        for i in range(len(sigmas) - 1):
            # For noise levels inside the range [s_tmin, s_tmax], we add more noise to the sample before removing noise.
            # We clamp gamma to never introduce more new noise than what is already present in the image - Karras et al.
            gamma = (
                min(self.s_churn / (len(sigmas) - 1), 2**0.5 - 1) if self.s_tmin <= sigmas[i] <= self.s_tmax else 0.0
            )
            eps = torch.randn_like(noisy_sample) * self.s_noise
            sigma_hat = sigmas[i] * (gamma + 1)
            # if gamma > 0, use additional noise level for computation ODE-> SDE Solver
            if gamma > 0:
                noisy_sample = noisy_sample + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
            denoised, step_info = model(
                noisy_sample, sigma_hat, conditions, extra_inputs, diffusion_step=i, return_info=return_steps
            )
            d = (noisy_sample - denoised) / sigma_hat
            dt = sigmas[i + 1] - sigma_hat
            # We apply a correction step to the Euler step to get a 2nd order approximation.
            # If we only are at the last step we use an Euler step for our update otherwise the heun one
            if sigmas[i + 1] == 0:
                # Euler method
                noisy_sample = noisy_sample + d * dt
            else:
                # Heun's method
                noisy_sample_2 = noisy_sample + d * dt
                denoised_2, step_info = model(
                    noisy_sample_2, sigmas[i + 1], conditions, extra_inputs, diffusion_step=i, return_info=return_steps
                )
                d_2 = (noisy_sample_2 - denoised_2) / sigmas[i + 1]
                d_prime = (d + d_2) / 2
                noisy_sample = noisy_sample + d_prime * dt

            # Append step info to info dictionary. All values in the info TensorDict are TensorDicts, except steps.
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
