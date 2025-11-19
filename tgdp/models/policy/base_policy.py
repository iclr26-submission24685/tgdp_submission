"""Base class for policies."""

from abc import ABC, abstractmethod
from typing import Optional

import torch
from tensordict import TensorDict


class BasePolicy(torch.nn.Module, ABC):
    """Base class for policy models.

    This abstract class defines the interface for policy models, including methods
    for computing actions and losses. Subclasses should implement the compute_action
    and loss methods.
    """

    def forward(self, observation: torch.Tensor, plan: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
        """Alias for compute_action."""
        return self.compute_action(observation, plan, t_idx)

    @abstractmethod
    def compute_action(self, observation: torch.Tensor, plan: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
        """Compute an action based on the given plan.

        This method should be overridden by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses")

    @abstractmethod
    def loss(
        self,
        batch: Optional[TensorDict] = None,
    ) -> torch.Tensor:
        """Compute the loss for the given batch.

        This method should be overridden by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses")
