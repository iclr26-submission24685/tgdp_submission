"""SelectActionPolicy class for selecting actions from a plan that already features actions."""

from typing import Optional

import torch
from tensordict import TensorDict

from .base_policy import BasePolicy


class SelectActionPolicy(BasePolicy):
    """A policy that selects the action at the given timestep.

    This assumes, that the action is the first part of the plan steps.
    """

    def __init__(
        self,
        action_dim: int,
    ):
        """Initialize the SelectActionPolicy.

        Args:
            action_dim (int): The number of dimensions of the action space.

        """
        super().__init__()
        self.action_dim = action_dim

    def compute_action(self, observation: torch.Tensor, plan: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
        """Select the action of the plan.

        Args:
            observation (torch.Tensor): The current observation to compute the action at.
            plan (torch.Tensor): The plan to compute the action on.
            t_idx (torch.Tensor): The time index to compute the action at.

        Returns:
            action (torch.Tensor): The computed action

        """
        assert torch.all(plan.shape[1] > t_idx), "Time index is out of bounds."
        return plan[torch.arange(plan.size(0)), t_idx, : self.action_dim]

    def loss(
        self,
        batch: Optional[TensorDict] = None,
    ) -> torch.Tensor:
        """Compute the loss for the given batch.

        The SelectActionPolicy is not trained, so this method should not be called.

        Args:
            sample: The sample to compute the loss on.
            sigma: The noise to use for the loss.
            conditions: The conditions to use for the loss.
            extra_inputs: The extra inputs to use for the loss.
            batch: The batch to compute the loss on.
            train_step: The current training step.

        Returns:
            loss: The computed loss.

        """
        raise NotImplementedError("This method should be overridden by subclasses")
