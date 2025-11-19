"""Planner that uses a diffusion model to generate plans of states and/or actions."""

import logging
from typing import Optional, Tuple

import torch
from tensordict import TensorDict

from tgdp import sigma_distributions
from tgdp.inpainting_conditioning import (
    ConditionType,
    apply_conditions,
    generate_observation_condition,
)
from tgdp.models.diffusion.base_diffusion import BaseDiffusionModel
from tgdp.models.planning.base_planner import BasePlanner
from tgdp.sampling.base_sampler import BaseSampler
from tgdp.sampling.noise_schedulers import BaseNoiseScheduler

logger = logging.getLogger(__name__)


class DiffusionPlanner(BasePlanner):
    """Diffusion-based planner for generating plans of states and/or actions."""

    def __init__(
        self,
        # Sampling and Noise Components.
        diffusion_model: BaseDiffusionModel,
        sampler: BaseSampler,
        sigma_distribution: sigma_distributions.BaseSigmaDistribution,
        noise_scheduler: BaseNoiseScheduler,
        diffusion_steps: int,
        # Sample dimensions.
        horizon: int,
        action_dim: int,
        observation_dim: int,
        plan_actions: bool,
        plan_observations: bool,
        # Inpainting condtitioning.
        observation_conditioning: bool = True,
        # Ensemble parameters.
        ensemble_num_samples: int = 1,
        ensemble_reduction: str = "max",
        ensemble_value_key: str = "trajectory_return",
    ) -> None:
        """Initialize the DiffusionPlanner with a diffusion model and a sampler."""
        assert plan_actions or plan_observations, "At least one of plan_actions or plan_observations must be True."
        assert plan_observations or not observation_conditioning, (
            "Observation condition inpainting is only valid when diffusing observations."
        )

        super().__init__(horizon, plan_observations, plan_actions)
        self.diffusion_model = diffusion_model

        # Sampling and Noise components
        self.sampler = sampler
        self.sigma_distribution = sigma_distribution
        self.noise_scheduler = noise_scheduler
        self.horizon = horizon

        # Parameters.
        self.diffusion_steps = diffusion_steps
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.sample_dimensions = (action_dim if plan_actions else 0) + (observation_dim if plan_observations else 0)

        # Conditioning parameters.
        self.observation_conditioning = observation_conditioning

        # Ensemble parameters.
        self.ensemble_num_samples = ensemble_num_samples
        self.ensemble_reduction = ensemble_reduction
        self.ensemble_value_key = ensemble_value_key

    def plan(
        self,
        observation: torch.Tensor,
    ) -> torch.Tensor:
        """Compute a plan for the given observation.

        This method generates an ensemble of candidate plans based on the provided observation.
        It then selects the best plan from the ensemble according to a specified reduction method.

        Args:
            observation: The current observation. (batch_size, observation_dim)

        Returns:
            plan: The plan for each observation. (batch_size, horizon, sample_dimensions)

        """
        # Diffuse an ensemble of plans.
        plan_ensemble, values, _ = self.compute_plan_ensemble(observation=observation)

        # Reduce the ensemble to a single plan.
        if self.ensemble_num_samples > 1 and values is not None:
            plan = self._reduce_plan_ensemble(plan_ensemble, values)
        else:
            plan = plan_ensemble

        return plan

    def compute_plan_ensemble(
        self,
        observation: torch.Tensor,
        num_samples: Optional[int] = None,
        return_steps: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[TensorDict]]:
        """Compute an ensemble of plans for the given observation .

        This method generates an ensemble of candidate plans based on the provided observation and,
        using the diffusion model and sampler.

        Args:
            observation: The current observation. (batch_size, observation_dim)
            num_samples: The number of samples to generate. If None, the number of samples is the ensemble_num_samples.
                Defaults to None.
            return_steps: If True, return the intermediate steps of the diffusion process. This includes gradients,
                intermediate samples and values.

        Returns:
            plan: The plan for each observation. (batch_size, horizon, sample_dimensions)
            Optional[value]: The value to optimize over for the given key.
            Optional[steps_info]: The steps_info TensorDict from the sampler.

        """
        # Set num_samples if none is provided, we use the ensemble parameter.
        if num_samples is None:
            num_samples = self.ensemble_num_samples

        # Prepare conditions.
        conditions = {}
        if self.observation_conditioning:
            current_time_idx = 0
            conditions = generate_observation_condition(
                current_time_idx,
                observation,
                conditions,
            )

        # Prepare extra inputs.
        extra_inputs = TensorDict({})
        extra_inputs["local_condition"] = TensorDict({})
        extra_inputs["global_condition"] = TensorDict({})
        extra_inputs = extra_inputs.to(observation.device)

        # Upsample the conditions and extra inputs for ensembling.
        if num_samples > 1:
            conditions, extra_inputs = self._prepare_ensemble_conditions_and_extra_inputs(
                conditions, extra_inputs, num_samples
            )

        # Get sigma schedule.
        sigmas = self.noise_scheduler.get_sigmas(
            self.diffusion_steps, device=observation.device, dtype=observation.dtype
        )

        # Initialize the sample.
        noisy_sample = (
            torch.randn(
                (observation.shape[0] * num_samples, self.horizon, self.sample_dimensions),
                device=observation.device,
            )
            * self.noise_scheduler.sigma_max
        )

        # Condition the sample.
        noisy_sample = apply_conditions(noisy_sample, conditions, (1 + self.noise_scheduler.sigma_max**2) ** 0.5)

        # Sample the trajectory ensemble.
        ensemble, step_infos = self.sampler.sample(
            self.diffusion_model,
            noisy_sample,
            sigmas,
            conditions,
            extra_inputs,
            return_steps,
        )

        # Compute values for the ensemble.
        if self.ensemble_value_key is not None:
            values = self.diffusion_model.compute_classifier_values(
                ensemble, sigmas[-1], conditions, extra_inputs, self.ensemble_value_key
            )
        else:
            values = None

        return ensemble, values, step_infos

    def _prepare_ensemble_conditions_and_extra_inputs(
        self,
        conditions: Optional[ConditionType],
        extra_inputs: Optional[TensorDict],
        num_samples: Optional[int] = None,
    ) -> Tuple[ConditionType, TensorDict]:
        """Prepare the conditions and extra inputs to diffuse an ensemble of plans as one batch.

        Args:
            conditions: The original conditions dictionary.
                {(time, start_idx, end_idx): condition[batch_size, condition_dim]}.
            extra_inputs: The original extra inputs to the model.
            num_samples: The number of samples to generate for the ensemble.

        Returns:
            conditions: The augmented conditions dictionary.
                {(time, start_idx, end_idx): condition[num_samples*batch_size, condition_dim]}.
            extra_inputs: The augmented extra inputs to the model.
                TensorDic{"local_condition": TensorDict, "global_condition": TensorDict}

        """
        # Set num_samples.
        if num_samples is None:
            num_samples = self.ensemble_num_samples

        if conditions is None:
            conditions = {}
        else:
            # Upsample conditions to the desired number of samples.
            for key, value in conditions.items():
                conditions[key] = value.repeat_interleave(num_samples, dim=0)

        if extra_inputs is None:
            extra_inputs = TensorDict.from_dict({"local_condition": TensorDict({}), "global_condition": TensorDict({})})
        else:
            # Upsample extra inputs to the desired number of samples.
            if extra_inputs is not None:
                for k1, v1 in extra_inputs.items():
                    if isinstance(v1, TensorDict):
                        td = TensorDict({})
                        for k2, v2 in v1.items():
                            if isinstance(v2, torch.Tensor):
                                td[k2] = v2.repeat_interleave(num_samples, dim=0)
                        extra_inputs[k1] = td
                    else:
                        raise ValueError(f"Unsupported type for extra input {k1}: {type(v1)}. Expected TensorDict.")

        return conditions, extra_inputs

    def _reduce_plan_ensemble(
        self,
        plans: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        """Select the trajectory with the highest value. Values should be stored in the values tensor.

        Select the best plan from the ensemble of plans based on the specified reduction method. The plans are of
        shape (batch_size * num_samples, time, features) and the values are of shape
        (batch_size * num_samples). The method supports different reduction strategies such as 'max' or 'min''.
        The 'max' strategy selects the plan with the highest value, while 'min' selects the one with the
        lowest value.This method is useful for ensemble-based planning where multiple candidate plans are generated,
        and the best one needs to be selected based on their values.

        Args:
            plans: The sample tensor. (batch_size * num_samples, time, features)
            values: The values tensor, containing values to optimize for. (batch_size * num_samples)

        Returns:
            plan: The best plan. (batch_size, time, features)

        """
        # Get best indices for all samples.
        if self.ensemble_reduction == "max":
            best_values, best_indices = torch.max(values.view(-1, self.ensemble_num_samples), dim=1)
        elif self.ensemble_reduction == "min":
            best_values, best_indices = torch.min(values.view(-1, self.ensemble_num_samples), dim=1)
        else:
            raise ValueError(f"Unknown ensemble_reduction method: {self.ensemble_reduction}")
        best_indices = self.ensemble_num_samples * torch.arange(len(best_indices), device=plans.device) + best_indices

        # Select and return the best samples.
        return plans[best_indices]

    def loss(self, batch: TensorDict) -> Tuple[torch.Tensor, TensorDict]:
        """Calculate the loss for the planner.

        Args:
            batch (TensorDict): Batch of data to calculate the loss on.

        Returns:
            TensorDict: The calculated loss. In this case, it returns an empty TensorDict.

        """
        # Prepare the sample.
        observations = batch["observations"] if self.plan_observations else None
        actions = batch["actions"] if self.plan_actions else None
        sample = torch.cat([i for i in [actions, observations] if i is not None], dim=2)

        # Sample sigmas.
        sigma = self.sigma_distribution.sample(shape=(len(sample), 1, 1), device=sample.device, dtype=sample.dtype)

        # Prepare conditions.
        conditions = {}
        if self.observation_conditioning and observations is not None:
            conditions = generate_observation_condition(0, observations[:, 0, :], conditions)

        # Prepare extra inputs.
        extra_inputs = TensorDict({})
        extra_inputs["local_condition"] = TensorDict({})
        extra_inputs["global_condition"] = TensorDict({})
        extra_inputs = extra_inputs.to(sample.device)

        # Call loss of the diffusion model.
        loss, loss_info = self.diffusion_model.loss(
            sample=sample,
            sigma=sigma,
            conditions=conditions,
            extra_inputs=extra_inputs,
            batch=batch,
        )
        return loss, loss_info
