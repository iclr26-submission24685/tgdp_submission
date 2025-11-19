"""Base class for trainers."""

from abc import ABC
from typing import Any, Dict, List

import lightning.pytorch as pl
import torch

from ..datasets import BaseDataset
from ..rendering import BaseRenderer


class BaseTrainer(pl.Trainer, ABC):
    """Base class for trainers."""

    dataset: BaseDataset
    renderer: BaseRenderer
    device: torch.device
    dtype: torch.dtype

    def __init__(self, *args, **kwargs):
        """Initialize the BaseTrainer."""
        super(BaseTrainer, self).__init__(*args, **kwargs)

    def train(self) -> None:
        """Train the model."""
        raise NotImplementedError("The train method must be implemented by the subclass.")

    def rollout_episodes(
        self,
        n_episodes: int = 1,
        max_episode_length: int = 1000,
        return_plans: bool = False,
        env_kwargs: Dict[str, Any] = {},
    ) -> List[Dict[str, Any]]:
        """Roll out episodes in the environment.

        Args:
            n_episodes: The number of episodes to roll out.
            max_episode_length: The maximum length of each episode.
            return_plans: If True, returns the plans for each episode.
            env_kwargs: Additional keyword arguments for the environment.

        Returns:
            List of dictionaries containing episode data.

        """
        raise NotImplementedError("The rollout_episodes method must be implemented by the subclass.")
