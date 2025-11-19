"""Base class for agents in the framework."""

from abc import ABC, abstractmethod
from typing import List, Optional, Union

import lightning.pytorch as pl
import numpy as np
import torch
from tensordict import TensorDict


class BaseAgent(pl.LightningModule, ABC):
    """Base class for agents."""

    def forward(
        self,
        observation: torch.Tensor,
    ) -> torch.Tensor:
        """Alias for act() method.

        Computes the action to take in the given observation. It also updates the history of the agent.

        Args:
            observation: The current observation. (batch_size, observation_dim)

        Returns:
            action: The action to take for each observation. (batch_size, action_dim)

        """
        return self.act(observation)

    def act(
        self,
        observation: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the action to take in the given observation."""
        raise NotImplementedError("The forward method must be implemented by the subclass.")

    @abstractmethod
    def training_step(self, batch: TensorDict, batch_idx) -> torch.Tensor:
        """Perform a single training step.

        Args:
            batch: A batch of data containing observations, actions, and rewards.
            batch_idx: The index of the batch in the current epoch.

        Returns:
            loss: The computed loss for the training step.

        """
        raise NotImplementedError("The train_step method must be implemented by the subclass.")

    @abstractmethod
    def configure_optimizers(self):
        """Configure the optimizers and learning rate schedulers for the agent."""
        raise NotImplementedError("The configure_optimizers method must be implemented by the subclass.")

    @abstractmethod
    def get_plan(self, idx: Optional[int] = None) -> TensorDict:
        """Get the current plan of the agent.

        Args:
            idx: The index of the batch to get the plan for. If None, returns the plan for all batches.

        Returns:
            plan: The current plan of the agent. If idx is None, returns a list of plans for all batches.

        """
        raise NotImplementedError("The get_plan method must be implemented by the subclass.")

    @abstractmethod
    def delete_episode(self, idx: Union[None, int, List[int], np.ndarray] = None) -> None:
        """Delete the history and existing plans of the agent.

        Delete the history and existing plans of the agent. This should be used only when we run a fixed number of
        episodes in parallel and we want to delete the history and plans of the episodes that have been completed.
        If the respective environment is reset, use reset_episode() instead.

        Args:
            idx: The index of the batch to delete the history for. If None, deletes the history for all batches.

        """
        raise NotImplementedError("The delete_episode method must be implemented by the subclass.")
