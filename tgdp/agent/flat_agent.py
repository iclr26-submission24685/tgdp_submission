"""FlatAgent module for planning and acting using diffusion models.

This module defines the FlatAgent class, which uses diffusion models and policies to predict and select actions
in a flat (non-hierarchical) manner.
"""

import logging
from functools import partial
from typing import List, Optional, Union

import numpy as np
import torch
from omegaconf import DictConfig
from tensordict import TensorDict

from ..models.planning.base_planner import BasePlanner
from ..models.policy.base_policy import BasePolicy
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

LOG_DICT_BUFFER_SIZE = 1000


class FlatAgent(BaseAgent):
    """A flat (no hierarchy) agent that uses diffusion to predict the trajectory and a policy to select the action."""

    def __init__(
        self,
        planner: BasePlanner,
        policy: BasePolicy,
        action_dim: int,
        observation_dim: int,
        replan_frequency: int = 1,
        optimizer: partial[torch.optim.Optimizer] = partial(torch.optim.Adam, lr=1e-4),
        lr_scheduler: Optional[partial[torch.optim.lr_scheduler.LRScheduler]] = None,
        stop_training_steps: Optional[dict] = None,
    ) -> None:
        """Initialize the FlatAgent class.

        Args:
            planner: The planner to be used for trajectory planning.
            policy: The policy to be used for action selection.
            action_dim: The number of action dimensions.
            observation_dim: The number of observation dimensions.
            learning_rate: The learning rate to use for training. Defaults to 1e-4.
            replan_frequency: The frequency at which to replan. This needs to be lower or equal to the horizon.
                Defaults to 1.
            optimizer: The optimizer to use for training. You can pass separate learning rates for different parameter
                groups by using a dictionary. This needs to be a partial object so that it can then be
                instantiated with the right parameters (e.g., functools.partial(...)). Defaults to torch.optim.Adam.
            lr_scheduler: The learning rate scheduler to use for training. This needs to be a partial object so that
                it can then be instantiated with the right parameters (e.g., functools.partial(...)). Defaults to None.
            stop_training_steps: A dictionary specifying the maximum number of training steps for different models.
                If None, no model will stop training. Example: {'planner': 10000}.
                Defaults to None.

        """
        super().__init__()

        # Model.
        self.planner = planner
        self.policy = policy
        self.trainable_models = {
            k: v
            for k, v in {
                "planner": planner,
                "policy": policy,
            }.items()
            if any(p.requires_grad for p in v.parameters())
        }

        # Stop training steps.
        if stop_training_steps is None:
            stop_training_steps = {}
        self.stop_training_steps = stop_training_steps

        # Sample dimensions.
        self.action_dim = action_dim
        self.observation_dim = observation_dim

        # Replanning.
        self.replan_frequency = replan_frequency
        self.replan_mask = None  # Stateful. Mask to determine which plans need to be recomputed.
        self.plan_step = None  # Stateful. Keeps track of the current step in the plan for each environment.
        self.plan = None  # Stateful. Keeps track of the plan per env.

        # Optimizer. If the lr is a dict, we use separate optimizers for different models.
        lr = optimizer.keywords["lr"]
        if isinstance(lr, DictConfig):
            assert "default" in lr, (
                "If you use a dict for the learning rate, you have to specify a default learning rate."
            )
            assert all(k in set(self.trainable_models.keys()) for k in lr if k != "default"), (
                f"You provided a learning rate for a model that does not exist or is not trainable. "
                f"Models: {set(self.trainable_models.keys())}, LR config: {lr}"
            )
            param_groups = [
                {"params": v.parameters(), "lr": lr[k], "name": k} for k, v in self.trainable_models.items() if k in lr
            ]
            self.optimizer = optimizer(param_groups, lr=lr["default"])
        else:
            self.optimizer = optimizer(self.parameters())

        # Learning rate scheduler. This instantiates the partial object.
        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler(self.optimizer)

    ############################################# Planning and Acting ##############################################

    def act(
        self,
        observation: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the next action for the agent based on the current observation.

        Computes the action, given the observation.

        Args:
            observation: The current observation. (batch_size, observation_dim)

        Returns:
            action: The action to take for each observation. (batch_size, action_dim)

        """
        # Checking for invalid inputs.
        assert observation.ndim == 2 and observation.shape[1] == self.observation_dim, (
            "Observation must be of shape [Batch, Observation Features]."
        )
        assert observation.device == next(self.parameters()).device, (
            "Observation must be on the same device as the model."
        )
        assert observation.dtype == next(self.parameters()).dtype, "Observation must be of the same dtype as the model."

        # Check if mode eval.
        if self.training:
            logger.warning("You are calling agent.act in training mode. You should probably call agent.eval() first.")

        # Increment the plan step.
        if self.plan_step is not None:
            self.plan_step += 1

        # For the first time we plan (after a run is started), all plans have to be computed.
        if self.plan is None or self.plan_step is None:
            self.plan_step = torch.tensor(0)
            self.plan = self.planner.plan(observation=observation)

        # Replan plans that are at the replan_frequency.
        elif self.plan_step == self.replan_frequency:
            self.plan_step = torch.tensor(0)
            self.plan = self.planner.plan(observation=observation)

        # Get the next action from the plan.
        t_idx = self.plan_step
        action = self.policy.compute_action(observation, self.plan, t_idx)

        return action

    ############################################## Training ##############################################

    def training_step(self, batch: TensorDict, batch_idx: int) -> torch.Tensor:
        """Perform a training step on the batch of data for all models.

        Args:
            batch: The batch of data to train on.
            batch_idx: The index of the batch.

        Returns:
            loss: The loss.

        """
        # Check if mode train.
        if not self.training:
            logger.warning(
                "You are calling agent.training_step in evaluation mode. You should probably call agent.train() first."
            )
        # Compute the losses.
        losses_sum = torch.zeros((), device=batch["observations"].device, dtype=batch["observations"].dtype)
        for key_model, v in self.trainable_models.items():
            if key_model not in self.stop_training_steps or self.stop_training_steps[key_model] > self.global_step:
                loss, loss_info = v.loss(
                    batch,
                )
                losses_sum += loss

                # Logging. We buffer the log dicts as lightning inbuilds are quite slow.
                loss_log_dict = loss_info["losses"].flatten_keys(separator="/").detach()
                info_log_dict = loss_info["losses_info"].flatten_keys(separator="/").detach()
                if not hasattr(self, "_log_dict_buffer"):
                    self._log_dict_buffer = {}
                else:
                    for key in loss_log_dict.keys():
                        key_flat = f"losses/{key_model}/{key}"
                        if key_flat in self._log_dict_buffer:
                            self._log_dict_buffer[key_flat] += loss_log_dict[key]
                        else:
                            self._log_dict_buffer[key_flat] = loss_log_dict[key]
                    for key in info_log_dict.keys():
                        key_flat = f"losses_info/{key_model}/{key}"
                        if key_flat in self._log_dict_buffer:
                            self._log_dict_buffer[key_flat] += info_log_dict[key]
                        else:
                            self._log_dict_buffer[key_flat] = info_log_dict[key]

        # Flush the log dict buffer if it exceeds the buffer size.
        if batch_idx > 0 and batch_idx % LOG_DICT_BUFFER_SIZE == 0:
            for key, value in self._log_dict_buffer.items():
                self.log(f"{key}", float(value) / LOG_DICT_BUFFER_SIZE, on_step=False, on_epoch=True)
            self._log_dict_buffer = {}

        return losses_sum

    def configure_optimizers(
        self,
    ) -> dict:
        """Return the optimizer and optionally lr scheduler for the model.

        Returns:
            optimizer: The optimizer to use for training the model.
            lr_scheduler: The learning rate scheduler to use for training the model. If None, no scheduler is used.

        """
        if hasattr(self, "lr_scheduler"):
            return {"optimizer": self.optimizer, "lr_scheduler": self.lr_scheduler}
        return {"optimizer": self.optimizer}

    ############################################## Stateful Agent ##############################################

    def get_plan(self, idx: Optional[int] = None) -> TensorDict:
        """Return the current plan of the agent as a TensorDict.

        This method returns the current plan of the agent. If idx is specified, it returns the plan for that specific
        environment. If idx is None, it returns the plan for all environments.
        The plan is a TensorDict with keys 'step', 'actions' and 'observations'. The 'actions' and 'observations'
        keys are only present if the respective diffusion is enabled. The 'step' key contains the current step of the
        plan, which is used to determine the current time step in the diffusion process.
        If the plan is not computed yet (i.e., plan_step is None), it returns an empty TensorDict.

        Args:
            idx: The index of the environment to return the plan for. If None, returns the plan for all environments.

        Returns:
            plan: The current plan. TensorDict with keys 'action', 'observation' and 'step'.

        """
        plan_dict = TensorDict({})
        if self.plan is not None and self.plan_step is not None:
            if self.planner.plan_actions:
                plan_dict["actions"] = (
                    self.plan[idx, :, : self.action_dim] if idx is not None else self.plan[:, :, : self.action_dim]
                )
            if self.planner.plan_observations:
                plan_dict["observations"] = (
                    self.plan[idx, :, -self.observation_dim :]
                    if idx is not None
                    else self.plan[:, :, -self.observation_dim :]
                )
            plan_dict["step"] = self.plan_step
        return plan_dict

    def delete_episode(self, idx: Union[None, int, List[int], np.ndarray] = None) -> None:
        """Delete the history and existing plans of the agent.

        Delete the history and existing plans of the agent. This should be used only when we run a fixed number of
        episodes in parallel and we want to delete the history and plans of the episodes that have been completed.
        If the respective environment is reset, use reset_episode() instead.

        Args:
            idx: The index of the batch to delete the history for. If None, deletes the history for all batches.

        """
        # If this is called before the first act(), we don't have to do anything.
        if self.plan_step is None or self.plan is None:
            return

        # Check if the indices are inside the bounds.
        if isinstance(idx, int):
            assert idx < len(self.plan), "The index is out of bounds."
        elif isinstance(idx, list):
            assert all(i < len(self.plan) for i in idx), "The index is out of bounds."

        # Delete plans.
        if idx is None:
            self.plan = None
        else:
            delete_mask = torch.ones(len(self.plan), dtype=torch.bool)
            delete_mask[idx] = False
            self.plan = self.plan[delete_mask]
