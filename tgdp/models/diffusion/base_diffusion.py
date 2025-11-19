"""Base class for diffusion models."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union, overload

import torch
from tensordict import TensorDict

from ...inpainting_conditioning import ConditionType


class BaseDiffusionModel(torch.nn.Module, ABC):
    """Base class for diffusion models."""

    @abstractmethod
    def forward(
        self,
        noisy_sample: torch.Tensor,
        sigma: torch.Tensor,
        conditions: Optional[ConditionType],
        extra_inputs: Optional[TensorDict],
        diffusion_step: int,
        return_info: bool = False,
    ) -> Tuple[torch.Tensor, Optional[TensorDict]]:
        """Perform a forward pass of the diffusion model.

        Args:
            noisy_sample: The noisy input sample tensor.
            sigma: The sigma value tensor.
            conditions: A dictionary of conditions for the model.
            extra_inputs: Additional inputs for the model.
            diffusion_step: The diffusion step from 0 to T-1.
            return_info: Whether to return additional information.

        Returns:
            A tuple containing the output tensor and a TensorDict with additional information.

        """
        raise NotImplementedError

    @abstractmethod
    def loss(
        self,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        conditions: Optional[ConditionType],
        extra_inputs: Optional[TensorDict],
        batch: TensorDict,
    ) -> Tuple[torch.Tensor, TensorDict]:
        """Compute the loss for the diffusion model.

        Args:
            sample (torch.Tensor): The input sample tensor.
            sigma (torch.Tensor): The sigma value tensor.
            conditions (Optional[ConditionType]): A dictionary of conditions for the model.
            extra_inputs (Optional[TensorDict]): Additional inputs for the model.
            batch (TensorDict): Optional batch data as a TensorDict.

        Returns:
            Tuple(torch.Tensor, TensorDict): The loss value and additional information.
                - loss: The loss value.
                - infos: TensorDict containing additional information.

        """
        raise NotImplementedError

    @overload
    def compute_classifier_values(
        self,
        noisy_sample: torch.Tensor,
        sigma: torch.Tensor,
        conditions: Optional[ConditionType],
        extra_inputs: Optional[TensorDict],
        key: str,
    ) -> torch.Tensor:
        raise NotImplementedError

    @overload
    def compute_classifier_values(
        self,
        noisy_sample: torch.Tensor,
        sigma: torch.Tensor,
        conditions: Optional[ConditionType],
        extra_inputs: Optional[TensorDict],
    ) -> TensorDict:
        raise NotImplementedError

    @abstractmethod
    def compute_classifier_values(
        self,
        noisy_sample: torch.Tensor,
        sigma: torch.Tensor,
        conditions: Optional[ConditionType],
        extra_inputs: Optional[TensorDict],
        key: Optional[str] = None,
    ) -> Union[torch.Tensor, TensorDict]:
        """Compute classifier values for the given noisy sample, sigma, and conditions.

        Args:
            noisy_sample: The noisy input sample tensor.
            sigma: The sigma value tensor.
            conditions: A dictionary of conditions for the model.
            extra_inputs: Additional inputs for the model.
            key: Optional key to specify which classifier value to compute.

        Returns:
            Either a tensor or a TensorDict containing the computed classifier values.

        """
        raise NotImplementedError
