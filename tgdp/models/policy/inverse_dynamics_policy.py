"""Inverse dynamics policy for predicting actions based on current and next observations."""

from typing import Optional, Tuple

import torch
from tensordict import TensorDict

from tgdp.models.diffusion.base_diffusion import BaseDiffusionModel

from ...networks.mlp.base_mlp_net import BaseMLPNet
from ...sampling.base_sampler import BaseSampler
from ...sampling.noise_schedulers import BaseNoiseScheduler
from ...sigma_distributions import BaseSigmaDistribution
from ..losses import PolicyL2, PolicyLoss
from .base_policy import BasePolicy


class InverseDynamicsPolicy(BasePolicy):
    """A policy that predicts the action given the current and next observation.

    This policy uses a feed-forward network to predict the action. The action is predicted based on the current
    observation and the next planned observation. The policy can be used in an open loop or closed loop setting.
    In the open loop setting, the action is predicted based on the current and next planned observation.
    In the closed loop setting, the action is predicted based on the current and the next planned observation.
    """

    def __init__(
        self,
        network: BaseMLPNet,
        observation_dim: int,
        action_dim: int,
        open_loop: bool = False,
        loss_fn: PolicyLoss = PolicyL2(),
    ):
        """Initialize the InverseDynamicsPolicy.

        Args:
            network (BaseInvDynamicsPolicyNet): The network to use for the policy.
            observation_dim (int): The dimension of the observation space.
            action_dim (int): The dimension of the action space.
            open_loop (bool): Whether the policy is open loop. Closed loop predicts an action based on the current
                observation and the next planned one. Open Loop predicts an action based on the current and next planned
                observation (the current observation is not taken into account).
            loss_fn (PolicyLoss): The loss function to use for the policy.

        """
        super().__init__()
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.open_loop = open_loop
        self.network = network
        self.loss_fn = loss_fn

    def compute_action(self, observation: torch.Tensor, plan: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
        """Compute the action given the current and next observation.

        Args:
            observation (torch.Tensor): The current observation to compute the action at.
            plan (torch.Tensor): The plan to compute the action on.
            t_idx (int): The time index to compute the action at.

        Returns:
            action (torch.Tensor): The computed action

        """
        assert (plan.shape[1] > t_idx + 1).all(), (
            "Time index is out of bounds. For the inverse dynamics policy, the planning horizon has to be at least the"
            "replan_frequency+1."
        )
        if self.open_loop:
            obs_t = plan[torch.arange(plan.size(0)), t_idx, -self.observation_dim :]
        else:
            obs_t = observation
        obs_t_plus_1 = plan[torch.arange(plan.size(0)), t_idx + 1, -self.observation_dim :]
        act_t = self.network(torch.concat([obs_t, obs_t_plus_1], dim=-1))
        return act_t

    def loss(
        self,
        batch: Optional[TensorDict] = None,
    ) -> Tuple[torch.Tensor, TensorDict]:
        """Compute the loss for the given batch.

        Args:
            batch (Optional[TensorDict]): The batch to compute the loss on.

        Returns:
            loss (torch.Tensor): The computed loss.

        """
        assert batch is not None, "Batch must be provided for the loss computation."

        # Prepare the sample.
        observations_t = batch["observations"][:, 0, -self.observation_dim :]
        observations_tp1 = batch["observations"][:, 1, -self.observation_dim :]
        target_actions = batch["actions"][:, 0, :]

        # Compute the predicted actions.
        predicted_actions = self.network(torch.concat([observations_t, observations_tp1], dim=-1))

        # Compute the loss.
        loss, info = self.loss_fn(predicted_actions, target_actions)

        return loss, TensorDict.from_dict({"losses": {"policy": loss}, "losses_info": info})


class DiffusionInverseDynamicsPolicy(BasePolicy):
    """A policy that predicts the action given the current and next observation using a diffusion model.

    This policy uses a diffusion model to predict the action. The action is predicted based on the current observation
    and the next planned observation. The policy can be used in an open loop or closed loop setting.
    In the open loop setting, the action is predicted based on the current and next planned observation.
    In the closed loop setting, the action is predicted based on the current and next planned observation.
    """

    def __init__(
        self,
        diffusion_model: BaseDiffusionModel,
        sampler: BaseSampler,
        noise_scheduler: BaseNoiseScheduler,
        sigma_distribution: BaseSigmaDistribution,
        diffusion_steps: int,
        observation_dim: int,
        action_dim: int,
        open_loop: bool = False,
        delta_obs: bool = False,
    ):
        """Initialize the DiffusionInverseDynamicsPolicy.

        Args:
            diffusion_model (BaseDiffusionModel): The diffusion model to use for the policy.
            sampler (BaseSampler): The sampler to use for the policy.
            noise_scheduler (BaseNoiseScheduler): The noise scheduler to use for the policy.
            sigma_distribution (BaseSigmaDistribution): The sigma distribution to use for the policy.
            diffusion_steps (int): The number of diffusion steps to use for the policy.
            observation_dim (int): The dimension of the observation space.
            action_dim (int): The dimension of the action space.
            open_loop (bool): Whether the policy is open loop or not. Closed loop predicts actions based on the current
                observation and the next planned one. Open Loop predicts actions based on the current and next planned
                observation (the current observation is not taken into account).
            delta_obs (bool): Whether to use delta observations or not. If True, the policy predicts the action based on
              the difference between the current and next observation. For the current obs 0 will be provided.

        """
        super().__init__()

        self.diffusion_model = diffusion_model

        # Diffusion components
        self.sampler = sampler
        self.noise_scheduler = noise_scheduler
        self.sigma_distribution = sigma_distribution
        self.diffusion_steps = diffusion_steps

        # Policy components
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.open_loop = open_loop
        self.delta_obs = delta_obs

    def compute_action(self, observation: torch.Tensor, plan: torch.Tensor, t_idx: torch.Tensor):
        """Compute the action given the current and next observation.

        Args:
            observation (torch.Tensor): The current observation to compute the action at.
            plan (torch.Tensor): The plan to compute the action on.
            t_idx (torch.Tensor): The time index to compute the action at.

        Returns:
            action (torch.Tensor): The computed action

        """
        assert (plan.shape[1] > t_idx + 1).all(), (
            "Time index is out of bounds. For the inverse dynamics policy, the planning horizon has to be at least "
            "the replan_frequency+1."
        )
        # Prepare observation condition.
        if self.open_loop:
            obs_t = plan[torch.arange(plan.size(0)), t_idx, -self.observation_dim :]
        else:
            obs_t = observation
        obs_t_plus_1 = plan[torch.arange(plan.size(0)), t_idx + 1, -self.observation_dim :]
        if self.delta_obs:
            obs_t_plus_1 = obs_t_plus_1 - obs_t
            obs_t = torch.zeros_like(obs_t)
        extra_inputs = TensorDict.from_dict(
            {"global_condition": TensorDict({"observation_t": obs_t, "observation_t_plus_1": obs_t_plus_1})}
        )

        # Get sigma schedule.
        sigmas = self.noise_scheduler.get_sigmas(
            self.diffusion_steps, device=observation.device, dtype=observation.dtype
        )

        # Initialize the sample.
        noisy_sample = (
            torch.randn((observation.shape[0], self.action_dim), device=observation.device)
            * self.noise_scheduler.sigma_max
        )

        # Sample the action using the diffusion model.
        action, info = self.sampler.sample(
            self.diffusion_model,
            noisy_sample,
            sigmas,
            conditions=None,
            extra_inputs=extra_inputs,
            return_steps=False,
        )

        return action

    def loss(self, batch: TensorDict) -> Tuple[torch.Tensor, TensorDict]:
        """Calculate the loss for the policy.

        Args:
            batch (TensorDict): Batch of data to calculate the loss on.

        Returns:
            TensorDict: The calculated loss. In this case, it returns an empty TensorDict.

        """
        # Prepare the sample.
        observations_t = batch["observations"][:, 0, -self.observation_dim :]
        observations_tp1 = batch["observations"][:, 1, -self.observation_dim :]
        target_actions = batch["actions"][:, 0, :]

        # Sample sigmas.
        sigma = self.sigma_distribution.sample(
            shape=(len(observations_t), 1), device=observations_t.device, dtype=observations_t.dtype
        )

        # Prepare extra inputs.
        if self.delta_obs:
            obs_t = torch.zeros_like(observations_t)
            obs_t_plus_1 = observations_tp1 - observations_t
        else:
            obs_t = observations_t
            obs_t_plus_1 = observations_tp1
        extra_inputs = TensorDict({})
        extra_inputs["local_condition"] = TensorDict({})
        extra_inputs["global_condition"] = TensorDict(
            {
                "obs_t": obs_t,
                "obs_t_plus_1": obs_t_plus_1,
            }
        )

        # Call loss of the diffusion model.
        loss, loss_info = self.diffusion_model.loss(
            sample=target_actions,
            sigma=sigma,
            conditions=None,
            extra_inputs=extra_inputs,
            batch=batch,
        )
        return loss, loss_info
