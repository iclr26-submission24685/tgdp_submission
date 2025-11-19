"""Base class for samplers in diffusion models."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
from tensordict import TensorDict

from ..inpainting_conditioning import ConditionType
from ..models.diffusion.base_diffusion import BaseDiffusionModel


class BaseSampler(ABC):
    """Base class for samplers."""

    @abstractmethod
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
            model: Model to be used for sampling
            noisy_sample: Initial noisy sample to be denoised
            sigmas: Sigma values to iterate over during sampling
            conditions: Optional dictionary of conditioning information for the model
            extra_inputs: Extra inputs to the model
            return_steps: If True, return the intermediate steps of the diffusion process.
                This includes gradients, intermediate samples and values

        Returns:
            Denoised sample.

        """
        raise NotImplementedError
