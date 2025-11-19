"""Base Classifier Network."""

from abc import ABC, abstractmethod
from typing import Optional

import torch
from tensordict import TensorDict


class BaseGuideNet(torch.nn.Module, ABC):
    """Abstract base class for (noise-conditional) classifier networks.

    These provide the same interface as diffusion models and are meant to be used as classifiers in diffusion models,
    for example as a guide in classifier-guidance.
    """

    @abstractmethod
    def forward(
        self,
        noisy_sample: torch.Tensor,
        sigma: torch.Tensor,
        extra_inputs: Optional[TensorDict] = None,
    ) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            noisy_sample (torch.Tensor): Noised sample as input to the model. (batch_size, horizon, features)
            sigma (torch.Tensor): Sigma as condition for the model. (batch_size, 1, 1) or ()
            extra_inputs (tensordict.TensorDict):
                - TensorDict with key "local_condition" and tensor value (batch_size, horizon, local_condition_size).
                - TensorDict with key "global_condition" and tensor value (batch_size,global_condition_size).

        Returns:
            torch.Tensor: Model output. (batch_size, horizon) if predict_one_value_per_timestep else (batch, 1).

        """
        raise NotImplementedError
