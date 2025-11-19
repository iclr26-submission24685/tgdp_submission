"""Base class for planners."""

from abc import ABC, abstractmethod
from typing import List, Optional

import torch
from tensordict import TensorDict


class BasePlanner(torch.nn.Module, ABC):
    """Base class for planners."""

    def __init__(
        self,
        horizon: int,
        plan_observations: bool = True,
        plan_actions: bool = True,
    ):
        """Initialize the BasePlanner."""
        super().__init__()
        self.horizon = horizon
        self.plan_observations = plan_observations
        self.plan_actions = plan_actions

    @abstractmethod
    def plan(
        self,
        observation: torch.Tensor,
        observation_history: Optional[List[torch.Tensor]] = None,
        action_history: Optional[List[torch.Tensor]] = None,
        goal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute a plan for the given observation and optional history and goal.

        This method generates an ensemble of candidate plans based on the provided observation and, optionally, a goal.
        It then selects the best plan from the ensemble according to a specified reduction method. The function supports
        both raw and latent observations, and can handle goal-conditioned planning if a goal is provided.

        Args:
            observation: The current observation (in latent space). (batch_size, observation_dim)
            observation_history: Optional history of observations. (batch_size, history_length, observation_dim)
            action_history: Optional history of actions. (batch_size, history_length, action_dim)
            goal: The goal in original space.
                It will be encoded if we use goal inpainting. (batch_size, observation_dim)

        Returns:
            plan: The plan for each observation. (batch_size, horizon, sample_dimensions)

        """

    @abstractmethod
    def loss(self, batch: TensorDict) -> TensorDict:
        """Calculate the loss for the planner.

        Args:
            batch (TensorDict): Batch of data to calculate the loss on.

        Returns:
            TensorDict: The calculated loss.

        """
        raise NotImplementedError("Loss method must be implemented by subclasses.")
