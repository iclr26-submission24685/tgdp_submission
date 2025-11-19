"""Contains callbacks for collecting data from the environment and testing the agent."""

import logging
from os import path
from typing import Optional, cast

import lightning.pytorch as pl
import numpy as np
import wandb
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

from ...agent.base_agent import BaseAgent
from ..base_trainer import BaseTrainer

logger = logging.getLogger(__name__)

DEFAULT_LOG_NORMALIZED = True


class TestAgentCallback(pl.Callback):
    """Test the agent on the environment and log the results. New episodes are not added to the dataset."""

    def __init__(
        self,
        num_episodes: int,
        test_every_n_epochs: Optional[int] = None,
        start_test_step: int = 0,
        render: bool = False,
        render_every_n_steps: Optional[int] = None,
        run_dir: Optional[str] = None,
    ):
        """Initialize the TestAgentCallback.

        Args:
            env (gym.Env): The environment to test the agent on.
            num_episodes (int): The number of episodes to test the agent on.
            test_every_n_epochs (int): The number of epochs between testing the agent.
            start_test_step (int): The step at which to start testing the agent.
            render (bool): Whether to render the environment.
            render_every_n_steps (int): The number of steps between rendering the environment.
            run_dir (str, optional): The directory for saving rollout videos.

        """
        self.num_episodes = num_episodes
        self.test_every_n_epochs = test_every_n_epochs
        self.start_test_step = start_test_step
        self.render = render
        self.render_every_n_steps = render_every_n_steps
        self.render_dir = path.join(run_dir, "rollouts") if run_dir is not None else None

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *_):
        """Call at the end of every training epoch."""
        # Casting to required subclasses.
        trainer = cast(BaseTrainer, trainer)
        pl_module = cast(BaseAgent, pl_module)

        if (
            trainer.global_step >= self.start_test_step
            and self.test_every_n_epochs is not None
            and (trainer.current_epoch + 1) % self.test_every_n_epochs == 0
        ):
            # Set the agent to evaluation mode
            pl_module.eval()

            # Run the episodes
            episodes = trainer.rollout_episodes(self.num_episodes)

            # Log the results
            rewards = np.array([ep["rewards"].sum() for ep in episodes])
            episode_lengths = np.array([len(ep["rewards"]) for ep in episodes])

            # Normalize if necessary.
            if DEFAULT_LOG_NORMALIZED:
                norm_env = trainer.dataset.make_env()
                if hasattr(norm_env, "get_normalized_score"):
                    rewards = 100 * norm_env.get_normalized_score(np.array(rewards))  # type: ignore

            if trainer.logger is not None:
                trainer.logger.log_metrics(
                    {
                        "episodes/episode_reward_mean": float(rewards.mean()),
                        "episodes/episode_length_mean": float(episode_lengths.mean()),
                        "episodes/episode_dones": float(len(episode_lengths)),
                    },
                    step=trainer.global_step,
                )
            # Log histograms if the logger supports this.
            if isinstance(trainer.logger, TensorBoardLogger):
                trainer.logger.experiment.add_histogram(
                    "episodes/episode_rewards", rewards, global_step=trainer.global_step
                )
                trainer.logger.experiment.add_histogram(
                    "episodes/episode_lengths", episode_lengths, global_step=trainer.global_step
                )
            elif isinstance(trainer.logger, WandbLogger):
                table = wandb.Table(data=np.stack([rewards, episode_lengths]).T, columns=["reward", "length"])
                wandb.log({"episodes/episode_stats": table}, step=trainer.global_step)
            # Render a random episode
            if self.render:
                ind = np.random.randint(self.num_episodes)
                trainer.renderer.render_rollout(
                    rollout=episodes[ind],
                    file_name=f"rollout-step={trainer.global_step}-reward{rewards[ind]}.mp4",
                )

            # Put the model back in training mode
            pl_module.train()
