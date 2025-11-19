"""DDIM Sampler."""

# Adapted from Scheikl et. al. - Movement Primitive Diffusion
# https://github.com/ScheiklP/movement-primitive-diffusion/blob/main/movement_primitive_diffusion/samplers/ddim.py

import logging
from typing import Optional, Tuple

import torch
from tensordict import TensorDict

from ..inpainting_conditioning import ConditionType
from ..models.diffusion.base_diffusion import BaseDiffusionModel
from .base_sampler import BaseSampler

logger = logging.getLogger(__name__)


class DDIMSampler(BaseSampler):
    """DPM-Solver 1 (equivalent to DDIM sampler)."""

    def __init__(self, temperature: float = 1.0):
        """Initialize the DDIMSampler.

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
        """Sample loop to calculate the noise free sample.

        Implements the DDIM sampling algorithm, which is a deterministic sampling method for diffusion models.
        It iteratively denoises the noisy sample using the provided diffusion model and sigma values.

        Args:
            model: Diffusion model to be used for sampling.
            noisy_sample: Initial noisy sample. (batch_size, time, features)
            sigmas: Sigma values to iterate over during sampling. (diffusion_steps)
            conditions: Dictionary containing the conditions.
            extra_inputs: Extra inputs to the model.
            return_steps: If True, return the intermediate steps of the diffusion process. This includes gradients,
                intermediate samples and values.

        Returns:
            Denoised sample. (batch_size, time, features)
            infos: Dictionary containing additional information.
                - Steps: Intermediate noisy steps. [diffusion_steps, batch_size, time, features]
                - Gradients: Dict of gradients at each step. {guide: [diffusion_steps, batch_size, time, features]}
                - Diffusion targets: Dict of diffusion targets at each step.
                {diffusion_model: [diffusion_steps, batch_size, time, features]}
                If return_steps is False, this is an empty TensorDict.

        """
        # Adjust the initial sample to the temperature.
        noisy_sample = self.temperature * noisy_sample

        # Initialize dictionary for steps info.
        if return_steps:
            infos = TensorDict({"steps": torch.empty((len(sigmas), *noisy_sample.shape), device=noisy_sample.device)})
            infos["steps"][0] = noisy_sample
        else:
            infos = TensorDict({})

        # Define the time function.
        def t_fn(sigma):
            return sigma.log().neg()

        # Iterate over all sigmas.
        for i in range(len(sigmas) - 1):
            # Predict the next sample.
            denoised, step_info = model(
                noisy_sample, sigmas[i], conditions, extra_inputs, diffusion_step=i, return_info=return_steps
            )
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            h = t_next - t
            noisy_sample = (sigmas[i + 1] / sigmas[i]) * noisy_sample - (-h).expm1() * denoised

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
