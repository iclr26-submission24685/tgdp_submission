"""Implementations for various loss functions used in different models.

This module provides base classes for defining loss functions, including diffusion losses, value losses, and policy
losses.
Adapted from Janner et. al. - Diffuser
https://github.com/jannerm/diffuser/blob/main/diffuser/models/helpers.py
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from tensordict import TensorDict

logger = logging.getLogger(__name__)


class BaseLoss(torch.nn.Module, ABC):
    """Abstract base class for all loss functions used in the models.

    This class defines the interface for loss computation, requiring subclasses to implement the forward method.
    """

    @abstractmethod
    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, TensorDict]:
        """Compute the forward pass for the loss function."""
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        """Call the forward method to compute the loss.

        Args:
            *args: Positional arguments for the forward method.
            **kwargs: Keyword arguments for the forward method.

        Returns:
           Tuple(torch.Tensor, TensorDict): A tuple containing:
                - loss (torch.Tensor): The computed loss value.
                - info (TensorDict): A dictionary containing additional information about the loss.

        """
        return self.forward(*args, **kwargs)


class DiffusionLoss(BaseLoss):
    """Base class for diffusion losses."""

    def forward(
        self,
        pred: torch.Tensor,
        targ: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, TensorDict]:
        """Compute the loss between the predicted and target output of the diffusion model.

        Args:
            pred (Tensor): The predicted values. (batch, time, features)
            targ (Tensor): The target values. (batch, time, features)
            sample_weights (Tensor): The per sample weights for the loss.
            If none, all samples are weighted equally. Defaults to None. (batch, 1, 1)

        Returns:
            tuple: A tuple containing:
                - loss (torch.Tensor): The loss. (1)
                - dict: A dictionary containing additional info about the loss.

        """
        loss = self._loss(pred, targ)
        if sample_weights is not None:
            assert sample_weights.ndim == pred.ndim == targ.ndim, "Weights must have the same shape as pred and targ."
            loss = loss * sample_weights
        loss = loss.mean()
        return loss, TensorDict({}).detach()

    @abstractmethod
    def _loss(self, pred: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
        """Compute the loss between the predicted and target values.

        Args:
            pred (torch.Tensor): The predicted values. (batch, time, features)
            targ (torch.Tensor): The target values. (batch, time, features)

        Returns:
            torch.Tensor: The computed loss. (batch, time, features)

        """
        raise NotImplementedError


class DiffusionL1(DiffusionLoss):
    """Computes the L1 (absolute difference) loss between predictions and targets for diffusion models."""

    def _loss(self, pred, targ):
        """Compute the absolute difference loss between predictions and targets.

        Args:
            pred (torch.Tensor): The predicted values. (batch, time, features)
            targ (torch.Tensor): The target values. (batch, time, features)

        Returns:
            torch.Tensor: The computed L1 loss. (batch, time, features)

        """
        return torch.abs(pred - targ)


class DiffusionL2(DiffusionLoss):
    """Computes the mean squared error (MSE) loss between predictions and targets for diffusion models."""

    def _loss(self, pred, targ):
        """Compute the mean squared error (MSE) loss between the predicted and target values.

        Args:
            pred (torch.Tensor): The predicted values. (batch, time, features)
            targ (torch.Tensor): The target values. (batch, time, features)

        Returns:
            torch.Tensor: The computed MSE loss. (batch, time, features)

        """
        return F.mse_loss(pred, targ, reduction="none")


class WeightedDiffusionLoss(DiffusionLoss):
    """Base class for weighted diffusion losses."""

    weights: torch.Tensor

    def __init__(
        self,
        action_dim: int,
        observation_dim: int,
        horizon: int,
        weight_config: Dict[str, Any] = {},
    ):
        """Initialize the weighted loss function, given the weight configuration.

        Args:
            action_dim (int): The dimension of the action space.
            observation_dim (int): The dimension of the observation space.
            horizon (int): The length of the trajectory.
            weight_config (dict): A dictionary containing the weight configuration. It may contain the following keys:
                - first_action_weight: float
                    Coefficient on first action loss
                - first_action_index: int
                    Index of first action in trajectory. This is equal to the length of the history that we condition
                    on. If None, defaults to 0.
                - weight_discount: float
                    Multiplies t^th timestep of trajectory loss by discount**t
                - action_weights: dict
                    {i: c} multiplies dimension i of observation loss by c
                - observation_weights: dict
                    {i: c} multiplies dimension i of observation loss

        """
        super().__init__()
        weights = self._compute_loss_weights(action_dim, observation_dim, horizon, weight_config)
        # Register the weights as a buffer so that they will be put on the right device via model.to().
        self.register_buffer("weights", weights)
        self.action_dim = action_dim
        self.observation_dim = observation_dim

    def _compute_loss_weights(
        self,
        action_dim: int,
        observation_dim: int,
        horizon: int,
        weight_config: Dict[str, Any] = {},
    ):
        """Generate the loss weights for the loss function, given the weight configuration.

        Args:
            weight_config (dict): A dictionary containing the weight configuration. It may contain the following keys:
                - first_action_weight: float
                    Coefficient on first action loss
                - first_action_index: int
                    Index of first action in trajectory. This is equal to the length of the history that we condition
                    on. If None, defaults to 0.
                - discount: float
                    Multiplies t^th timestep of trajectory loss by discount**t
                - action_weights: dict
                    {i: c} multiplies dimension i of observation loss by c
                - observation_weights: dict
                    {i: c} multiplies dimension i of observation loss by c
            action_dim (int): The dimension of the action space.
            observation_dim (int): The dimension of the observation space.
            horizon (int): The length of the trajectory.

        """
        # Initialize the weights tensor.
        weights = torch.ones(horizon, action_dim + observation_dim, dtype=torch.float32)

        # Set the weight for the first action.
        if "first_action_weight" in weight_config:
            first_action_weight = weight_config["first_action_weight"]
            first_action_index = weight_config.get("first_action_index", 0)
            logger.debug("No first_action_index provided. Defaulting to 0.")
            weights[first_action_index, :action_dim] = first_action_weight
            self.first_action_index = first_action_index

        if "next_observation_weight" in weight_config:
            next_observation_weight = weight_config["next_observation_weight"]
            next_observation_index = weight_config.get("next_observation_index", 1)
            logger.debug("No next_observation_index provided. Defaulting to 1.")
            weights[next_observation_index, -observation_dim:] *= next_observation_weight
            self.next_observation_index = next_observation_index

        # Apply action weights.
        if "action_weights" in weight_config:
            weights_dict = weight_config["action_weights"]
            for ind, w in weights_dict.items():
                assert ind < action_dim, f"Action index {ind} is out of bounds."
                weights[:, ind] *= w

        # Apply observation weights.
        if "observation_weights" in weight_config:
            weights_dict = weight_config["observation_weights"]
            for ind, w in weights_dict.items():
                assert ind < observation_dim, f"Observation index {ind} is out of bounds."
                weights[:, -observation_dim + ind] *= w

        # Discount the loss with the trajectory timestep.
        if "weight_discount" in weight_config:
            discount = weight_config["weight_discount"]
            weights = weights * discount ** torch.arange(horizon, dtype=torch.float32).unsqueeze(1)

        # Normalize the weights.
        weights = weights / weights.mean()

        return weights

    def forward(
        self,
        pred: torch.Tensor,
        targ: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, TensorDict]:
        """Compute the forward pass for the loss function.

        Args:
            pred (torch.Tensor): The predicted values. (batch, time, features)
            targ (torch.Tensor): The target values. (batch, time, features)
            sample_weights (torch.Tensor): The per sample weights for the loss.
            If none, all samples are weighted equally. Defaults to None. (batch, 1, 1)

        Returns:
            tuple: A tuple containing:
                - weighted_loss (torch.Tensor): The mean weighted loss. (1)
                - info(TensorDict): Contains additional info about the loss. This may include:
                    - "a0_loss" (torch.Tensor): The loss for the first action.
                    - "o1_loss" (torch.Tensor): The loss for the next observation.
                    - "action_loss" (torch.Tensor): The loss for all actions.
                    - "observation_loss" (torch.Tensor): The loss for all observations.

        """
        # Compute the weighted loss
        loss = self._loss(pred, targ)

        if sample_weights is not None:
            assert sample_weights.ndim == pred.ndim == targ.ndim, "Weights must have the same shape as pred and targ."
            weighted_loss = loss * self.weights * sample_weights
        else:
            weighted_loss = loss * self.weights

        # Compute loss infos.
        loss_info = TensorDict({})
        if self.action_dim > 0:
            loss_info["action_loss"] = (
                weighted_loss[:, :, : self.action_dim] / self.weights[:, : self.action_dim]
            ).mean()
        if self.observation_dim > 0:
            loss_info["observation_loss"] = (
                weighted_loss[:, :, -self.observation_dim :] / self.weights[:, -self.observation_dim :]
            ).mean()
        if hasattr(self, "first_action_index"):
            loss_info["a0_loss"] = (
                weighted_loss[:, self.first_action_index, : self.action_dim]
                / self.weights[self.first_action_index, : self.action_dim]
            ).mean()
        if hasattr(self, "next_observation_index"):
            loss_info["o1_loss"] = (
                weighted_loss[:, self.next_observation_index, -self.observation_dim :]
                / self.weights[self.next_observation_index, -self.observation_dim :]
            ).mean()

        return weighted_loss.mean(), loss_info.detach()


class WeightedDiffusionL1(WeightedDiffusionLoss):
    """Computes the weighted L1 (absolute difference) loss between predictions and targets for diffusion models."""

    def _loss(self, pred: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
        """Compute the absolute difference loss between predictions and targets.

        Args:
            pred (torch.Tensor): The predicted values. (batch, time, features)
            targ (torch.Tensor): The target values. (batch, time, features)

        Returns:
            torch.Tensor: The computed L1 loss. (batch, time, features)

        """
        return torch.abs(pred - targ)


class WeightedDiffusionL2(WeightedDiffusionLoss):
    """Computes the weighted mean squared error (MSE) loss between predictions and targets for diffusion models."""

    def _loss(self, pred: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
        """Compute the mean squared error (MSE) loss between the predicted and target values.

        Args:
            pred (torch.Tensor): The predicted values. (batch, time, features)
            targ (torch.Tensor): The target values. (batch, time, features)

        Returns:
            torch.Tensor: The computed MSE loss. (batch, time, features)

        """
        return F.mse_loss(pred, targ, reduction="none")


class ValueLoss(BaseLoss):
    """Base class for value losses."""

    def forward(self, pred, targ, sample_weights=None):
        """Compute the loss and additional statistics between predictions and targets.

        Args:
            pred (torch.Tensor): The predicted values. (batch, 1)
            targ (torch.Tensor): The target values. (batch, 1)
            sample_weights (torch.Tensor): The per sample weights for the loss. (batch, 1)
            If none, all samples are weighted equally. Defaults to None.

        Returns:
            tuple: A tuple containing:
                - loss (torch.Tensor): The computed loss value. (1)
                - info (dict): A dictionary containing additional statistics:
                    - "mean_pred" (torch.Tensor): Mean of the predicted values.
                    - "mean_targ" (torch.Tensor): Mean of the target values.
                    - "min_pred" (torch.Tensor): Minimum of the predicted values.
                    - "min_targ" (torch.Tensor): Minimum of the target values.
                    - "max_pred" (torch.Tensor): Maximum of the predicted values.
                    - "max_targ" (torch.Tensor): Maximum of the target values.
                    - "corr" (float or np.NaN): Correlation coefficient between predictions and targets if there are
                        multiple predictions, otherwise NaN.

        """
        if sample_weights is not None:
            assert sample_weights.ndim == pred.ndim == targ.ndim, "Weights must have the same shape as pred and targ."
            loss = (self._loss(pred, targ) * sample_weights).mean()
        else:
            loss = self._loss(pred, targ).mean()

        if len(pred) > 1 and len(pred.shape) <= 2:
            corr = torch.corrcoef(torch.stack([pred.squeeze(), targ.squeeze()]))[0, 1]
        else:
            corr = torch.nan

        info = {
            "mean_pred": pred.mean(),
            "mean_targ": targ.mean(),
            "min_pred": pred.min(),
            "min_targ": targ.min(),
            "max_pred": pred.max(),
            "max_targ": targ.max(),
            "corr": corr,
        }

        return loss, TensorDict(info).detach()

    @abstractmethod
    def _loss(self, pred: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
        """Compute the loss between the predicted and target values.

        Args:
            pred (torch.Tensor): The predicted values. (batch, 1)
            targ (torch.Tensor): The target values. (batch, 1)

        Returns:
            torch.Tensor: The computed loss. (batch, 1)

        """
        raise NotImplementedError("This method should be implemented by subclasses.")


class ValueL1(ValueLoss):
    """Computes the L1 (absolute difference) loss between predictions and targets for value models."""

    def _loss(self, pred: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
        """Compute the absolute difference loss between the predicted and target values.

        Args:
            pred (torch.Tensor): The predicted values. (batch, 1)
            targ (torch.Tensor): The target values. (batch, 1)

        Returns:
            torch.Tensor: The computed L1 loss. (batch, 1)

        """
        return torch.abs(pred - targ)


class ValueL2(ValueLoss):
    """Computes the mean squared error (MSE) loss between predictions and targets for value models."""

    def _loss(self, pred: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
        """Compute the mean squared error (MSE) loss between the predicted and target values.

        Args:
            pred (torch.Tensor): The predicted values. (batch, 1)
            targ (torch.Tensor): The target values. (batch, 1)

        Returns:
            torch.Tensor: The computed MSE loss. (batch, 1)

        """
        return F.mse_loss(pred, targ, reduction="none")


class ValueCE(ValueLoss):
    """Computes the binary cross-entropy loss between predictions and targets for value models."""

    def _loss(self, pred: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
        """Compute the binary cross-entropy loss between the predicted and target values.

        Args:
            pred (Tensor): The predicted values. (batch, 1)
            targ (Tensor): The target values. (batch, 1)

        Returns:
            Tensor: The computed binary cross-entropy loss. (batch, 1)

        """
        return F.binary_cross_entropy(pred, targ, reduction="none")


class PolicyLoss(BaseLoss):
    """Base class for policy losses."""

    def forward(
        self,
        pred: torch.Tensor,
        targ: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ):
        """Compute the loss between the predicted and target output of the policy.

        Args:
            pred (Tensor): The predicted values. (batch, action_dim)
            targ (Tensor): The target values. (batch, action_dim)
            sample_weights (Tensor): The per sample weights for the loss.
            If none, all samples are weighted equally. Defaults to None. (batch, 1)

        Returns:
            tuple: A tuple containing:
                - loss (torch.Tensor): The loss. (1)
                - dict: A dictionary containing additional info about the loss.

        """
        loss = self._loss(pred, targ)
        if sample_weights is not None:
            loss = loss * sample_weights
        loss = loss.mean()

        return loss, TensorDict({}).detach()

    @abstractmethod
    def _loss(self, pred: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
        """Compute the loss between the predicted and target values.

        Args:
            pred (torch.Tensor): The predicted values. (batch, action_dim)
            targ (torch.Tensor): The target values. (batch, action_dim)

        Returns:
            torch.Tensor: The computed loss. (batch, action_dim)

        """
        raise NotImplementedError("This method should be implemented by subclasses.")


class PolicyL1(PolicyLoss):
    """Computes the L1 (absolute difference) loss between predictions and targets for policy models."""

    def _loss(self, pred: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
        """Compute the absolute difference loss between predictions and targets.

        Args:
            pred (torch.Tensor): The predicted values. (batch, action_dim)
            targ (torch.Tensor): The target values. (batch, action_dim)

        Returns:
            torch.Tensor: The computed L1 loss. (batch, action_dim)

        """
        return torch.abs(pred - targ)


class PolicyL2(PolicyLoss):
    """Computes the mean squared error (MSE) loss between predictions and targets for policy models."""

    def _loss(self, pred: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
        """Compute the mean squared error (MSE) loss between the predicted and target values.

        Args:
            pred (torch.Tensor): The predicted values. (batch, action_dim)
            targ (torch.Tensor): The target values. (batch, action_dim)

        Returns:
            torch.Tensor: The computed MSE loss. (batch, action_dim)

        """
        return F.mse_loss(pred, targ, reduction="none")
