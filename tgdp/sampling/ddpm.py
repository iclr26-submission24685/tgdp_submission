"""DDPM Sampler (Denoising Diffusion Probabilistic Models)."""

import logging
from typing import Optional, Tuple

import torch
from tensordict import TensorDict

from ..inpainting_conditioning import ConditionType
from ..models.diffusion.base_diffusion import BaseDiffusionModel
from .base_sampler import BaseSampler

logger = logging.getLogger(__name__)


class DDPMSampler(BaseSampler):
    """DDPM sampler (EDM/Karras-style, stochastic), with temperature."""

    def __init__(self, temperature: float = 1.0):
        """Initialize the DDPMSimpleSampler with a specified temperature.

        Args:
            temperature: Temperature parameter for sampling as proposed in Ajay et al. The initial sample is multiplied
                by the value.

        """
        super().__init__()
        self.temperature = temperature

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
        """Generate a sample from the diffusion model using the DDPM sampling procedure.

        Args:
            model : BaseDiffusionModel
                The diffusion model used for denoising.
            noisy_sample : torch.Tensor
                The initial noisy input tensor.
            sigmas : torch.Tensor
                Sequence of noise levels (sigmas) for each diffusion step.
            conditions : Optional[ConditionType], optional
                Conditioning information for the model, by default None.
            extra_inputs : Optional[TensorDict], optional
                Additional inputs for the model, by default None.
            return_steps : bool, optional
                Whether to return intermediate steps and info, by default False.

        """
        # Adjust the initial sample to the temperature.
        noisy_sample = self.temperature * noisy_sample

        if return_steps:
            infos = TensorDict({"steps": torch.empty((len(sigmas), *noisy_sample.shape), device=noisy_sample.device)})
            infos["steps"][0] = noisy_sample
        else:
            infos = TensorDict({})

        # Iterate over all sigma values for DDPM updates.
        for i in range(len(sigmas) - 1):
            sigma_curr, sigma_next = sigmas[i], sigmas[i + 1]

            # Predict denoise via model.
            denoised, step_info = model(
                noisy_sample, sigma_curr, conditions, extra_inputs, diffusion_step=i, return_info=return_steps
            )

            # Compute mean (EDM/Karras version of ancestral DDPM step).
            mean = (sigma_next / sigma_curr) * noisy_sample + (1.0 - sigma_next / sigma_curr) * denoised

            # Variance is difference in noise scales squared. Avoid division by 0.
            var = torch.max(sigma_next**2 - sigma_curr**2, torch.tensor(1e-20, device=sigma_next.device))

            # Add random noise except last step.
            if i < len(sigmas) - 2:
                eps = torch.randn_like(noisy_sample)
                noisy_sample = mean + (var**0.5) * eps  # * self.temperature TODO Figure out what makes more sense.
            else:
                noisy_sample = mean

            if return_steps:
                for k1, v1 in step_info.items():
                    if i == 0:
                        infos[k1] = TensorDict(
                            {k2: torch.empty((len(sigmas) - 1, *v2.shape), device=v2.device) for k2, v2 in v1.items()}
                        )
                    for k2, v2 in v1.items():
                        infos[k1][k2][i] = v2
                infos["steps"][i + 1] = noisy_sample

        denoised_sample = noisy_sample
        return denoised_sample, infos.detach()
