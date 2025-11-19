"""Temperature-Guided Diffusion Model."""

import logging
from typing import Optional, Tuple, Union

import numpy as np
import torch
from omegaconf.dictconfig import DictConfig
from tensordict import TensorDict

from tgdp.networks.diffusion.base_diffusion_net import BaseDiffusionNet

from ...inpainting_conditioning import ConditionType, apply_conditions
from ...utils.extra_inputs import concat_extra_inputs
from ..guidance.base_guide import BaseGuide
from ..losses import DiffusionL2, DiffusionLoss
from ..scaling import EpsilonScaling, Scaling
from .base_diffusion import BaseDiffusionModel

logger = logging.getLogger(__name__)


class TemperatureGuidedDiffusion(BaseDiffusionModel):
    """Temperature-Guided Diffusion Model."""

    def __init__(
        self,
        network: BaseDiffusionNet,
        guides: Union[DictConfig, torch.nn.ModuleDict],
        guide_scale: float = 20.0,
        max_temperature: float = 3.0,
        target_temperature: float = 3.0,
        temperature_base: float = 0.0,
        guide_scale_alpha: float = 4.0,
        discrete_temperature_distribution: bool = False,
        unbiased_loss_probability: float = 0.25,
        batch_normalize_weights: bool = True,
        scaling: Scaling = EpsilonScaling(),
        loss_fn: DiffusionLoss = DiffusionL2(),
        conditioned_loss: bool = True,
        mask_conditioned_loss: bool = False,
    ):
        """Initialize the Temperature-Guided Diffusion Model.

        Args:
            network: Diffusion network.
            guides: Dictionary containing the classifier guidance networks.
            guide_scale: The scale of the conditioned diffusion target.
            max_temperature: Maximum temperature for the biased loss.
            target_temperature: Target temperature for sampling.
            temperature_base: Base temperature for the diffusion model.
            guide_scale_alpha: Alpha parameter for the guide scale that targets from different modes.
            discrete_temperature_distribution: Whether to use a discrete temperature distribution during training. If
                True, the temperature takes on discrete values in [-t_max, t_0, t_max].
            unbiased_loss_probability: Probability of a batch to be 0 temperature. If larger 0, the temperature is
                is sampled more often than others. This is especially true if we use discrete temperatures.
            batch_normalize_weights: Whether to batch normalize the sample weights during training.
            scaling: Scaling object for the diffusion model.
            loss_fn: Loss function for the diffusion model.
            conditioned_loss: Whether to condition the loss on the conditions.
            mask_conditioned_loss: Whether to mask the loss.

        """
        assert 0 <= unbiased_loss_probability <= 1, "Unbiased Loss probability must be in [0,1]."
        super().__init__()

        # Diffusion Network and Guides.
        self.network = network
        if guides is None:
            guides = {}
        self.guides = guides if isinstance(guides, torch.nn.ModuleDict) else torch.nn.ModuleDict(guides)

        # Guide scale and temperature parameters.
        self.guide_scale = guide_scale
        self.max_temperature = max_temperature
        self.target_temperature = target_temperature
        self.temperature_base = temperature_base
        self.guide_scale_alpha = guide_scale_alpha

        # Training temperature distribution.
        self.discrete_temperature_distribution = discrete_temperature_distribution
        self.unbiased_loss_probability = unbiased_loss_probability
        self.batch_normalize_weights = batch_normalize_weights

        # Scaling/preconditioning.
        self.scaling = scaling

        # Loss function parameters.
        self.loss_fn = loss_fn
        self.conditioned_loss = conditioned_loss
        self.mask_conditioned_loss = mask_conditioned_loss

    @torch.no_grad()
    def forward(
        self,
        noisy_sample: torch.Tensor,
        sigma: torch.Tensor,
        conditions: Optional[ConditionType],
        extra_inputs: Optional[TensorDict],
        diffusion_step: int,
        return_info: bool = False,
    ) -> Tuple[torch.Tensor, Optional[TensorDict]]:
        """Apply one step of classifier-free diffusion to the noisy sample.

        Args:
            noisy_sample: Noisy sample tensor. (batch_size, time, features)
            sigma: Noise level tensor. (batch_size, 1, 1).
            conditions: Dictionary containing the conditions.
            extra_inputs: Extra inputs dictionary.
            diffusion_step: Current diffusion step that goes from 0 to T-1.
            return_info: Whether to return additional information.

        Returns:
            denoised_sample: Denoised sample tensor (at t_0). (batch_size, time, features)
            infos: TensorDict containing additional information. If return_info is False, infos is None.

        """
        # Dictionary of additional infos.
        if return_info:
            info = TensorDict({})
        else:
            info = None

        # Compute the denoised sample.
        diffusion_targets = self._compute_diffusion_targets(
            noisy_sample, sigma, conditions, extra_inputs, diffusion_step
        )

        # Compute the denoised sample as adaptive combinations of the high and low conditional diffusion targets.
        if self.target_temperature != self.temperature_base and self.guide_scale > 0:
            adaptive_guide_scale = (
                self.guide_scale
                * (
                    torch.cosine_similarity(
                        (diffusion_targets["high-conditional"] - diffusion_targets["unconditional"]).view(
                            noisy_sample.shape[0], -1
                        ),
                        (diffusion_targets["unconditional"] - diffusion_targets["low-conditional"]).view(
                            noisy_sample.shape[0], -1
                        ),
                        dim=1,
                    )
                ).clip(min=-1, max=1)
                ** (self.guide_scale_alpha)
            ).view(-1, 1, 1)
            base = diffusion_targets["high-conditional"]
            grad = diffusion_targets["high-conditional"] - diffusion_targets["unconditional"]
            denoised_sample = base + adaptive_guide_scale * grad
        else:
            denoised_sample = diffusion_targets["high-conditional"]

        # Apply conditioning to the denoised sample.
        denoised_sample = apply_conditions(denoised_sample, conditions)

        # Add values, classifer gradients, and the diffusion target to the info dictionary.
        if return_info and info is not None:
            info["diffusion_targets"] = diffusion_targets

        return denoised_sample, info

    def _compute_diffusion_targets(
        self,
        noisy_sample: torch.Tensor,
        sigma: torch.Tensor,
        conditions: Optional[ConditionType],
        extra_inputs: Optional[TensorDict],
        diffusion_step: int,
    ) -> TensorDict:
        """Compute the diffusion target for the noisy sample.

        Args:
            noisy_sample: Noisy sample tensor. (batch_size, time, features)
            sigma: Noise level tensor. (batch_size, 1, 1) or ()
            conditions: Dictionary containing the conditions.
            extra_inputs: Extra inputs dictionary.
            diffusion_step: Current diffusion step that goes from 0 to T-1.

        Returns:
            denoised_sample: Denoised sample tensor (at t_0). (batch_size, time, features)

        """
        # Get scaling factors.
        c_skip, c_out, c_in, c_noise = self.scaling(sigma)

        # Scale and condition the noisy sample.
        noisy_sample = apply_conditions(noisy_sample, conditions, (1 + sigma**2).sqrt())
        network_input = c_in * noisy_sample

        # Compile the unconditional and conditional extra inputs.
        if extra_inputs is None:
            extra_inputs = TensorDict.from_dict({"global_condition": {}, "local_condition": {}})
        extra_inputs_high_conditional = extra_inputs.copy()
        extra_inputs_high_conditional["global_condition"]["temperature"] = (
            self.target_temperature
            / self.max_temperature
            * torch.ones((noisy_sample.shape[0], 1), device=noisy_sample.device)
        )
        extra_inputs_low_conditional = extra_inputs.copy()
        extra_inputs_low_conditional["global_condition"]["temperature"] = (
            -self.target_temperature
            / self.max_temperature
            * torch.ones((noisy_sample.shape[0], 1), device=noisy_sample.device)
        )
        extra_inputs_unconditional = extra_inputs.copy()
        extra_inputs_unconditional["global_condition"]["temperature"] = (
            self.temperature_base
            / self.max_temperature
            * torch.ones((noisy_sample.shape[0], 1), device=noisy_sample.device)
        )

        # Compute the denoising targets of the unconditional diffusion.
        denoised_sample_unconditional = self.network(
            network_input, c_noise, concat_extra_inputs(extra_inputs_unconditional)
        )
        denoised_sample_unconditional = c_skip * noisy_sample + c_out * denoised_sample_unconditional

        # Compute the denoising targets of the conditional diffusion.
        denoised_sample_high_conditional = self.network(
            network_input, c_noise, concat_extra_inputs(extra_inputs_high_conditional)
        )
        denoised_sample_high_conditional = c_skip * noisy_sample + c_out * denoised_sample_high_conditional

        # Compute the denoising targets of the conditional diffusion.
        denoised_sample_low_conditional = self.network(
            network_input, c_noise, concat_extra_inputs(extra_inputs_low_conditional)
        )
        denoised_sample_low_conditional = c_skip * noisy_sample + c_out * denoised_sample_low_conditional

        # Return the diffusion targets.
        diffusion_targets = TensorDict(
            {
                "unconditional": denoised_sample_unconditional,
                "high-conditional": denoised_sample_high_conditional,
                "low-conditional": denoised_sample_low_conditional,
            }
        )
        return diffusion_targets

    def compute_classifier_values(
        self,
        noisy_sample: torch.Tensor,
        sigma: torch.Tensor,
        conditions: Optional[ConditionType],
        extra_inputs: Optional[TensorDict],
        key: Optional[str] = None,
    ) -> Union[torch.Tensor, TensorDict]:
        """Compute the values of the guides.

        Args:
            noisy_sample: Noisy sample tensor. (batch_size, time, features)
            sigma: Noise level tensor. (batch_size, 1, 1) or ()
            conditions: Dictionary containing the conditions.
            extra_inputs: Extra inputs dictionary.
            key: Key of the guide to compute the value for. If None, all values are computed.

        Returns:
            values: Tensor containing the value of the classifier for the given key. If key is None a TensorDict with
                all values is returned.

        """
        # Get scaling factors
        _, _, c_in, c_noise = self.scaling(sigma)

        # Apply c_in scaling conditioning before predicting the value.
        noisy_sample_scaled_and_conditioned = apply_conditions(
            c_in * noisy_sample, conditions, c_in * (1 + sigma**2).sqrt()
        )

        # Compute the values of the guides.
        if key is not None:
            guide = self.guides[key]
            assert isinstance(guide, BaseGuide), "All guides must be instances of BaseGuide."
            value = guide.predict_value(noisy_sample_scaled_and_conditioned, c_noise, extra_inputs)
            return value
        else:
            values = TensorDict({})
            for guide_name, guide in self.guides.items():
                assert isinstance(guide, BaseGuide), "All guides must be instances of BaseGuide."
                # Compute the values of the guides.
                value = guide.predict_value(noisy_sample_scaled_and_conditioned, c_noise, extra_inputs)
                values[guide_name] = value

            return values

    def loss(
        self,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        conditions: Optional[ConditionType],
        extra_inputs: Optional[TensorDict],
        batch: TensorDict,
    ) -> Tuple[torch.Tensor, TensorDict]:
        """Compute the loss for the diffusion model.

        Args:
            sample (torch.Tensor): The input sample tensor.
            sigma (torch.Tensor): The sigma value tensor.
            conditions (Optional[ConditionType]): A dictionary of conditions for the model.
            extra_inputs (Optional[TensorDict]): Additional inputs for the model.
            batch (TensorDict): Optional batch data as a TensorDict.

        Returns:
            Tuple(torch.Tensor, TensorDict): The loss value and additional information.
                - loss: The loss value.
                - infos: TensorDict containing additional information.

        """
        # Dictionary of additional infos.
        infos = TensorDict.from_dict({"losses": {}, "losses_info": {}})

        # Initialize loss.
        loss = 0.0

        # Noise is first drawn from a normal distribution with mean 0 and std 1, then scaled by sigma (the desired std)
        noise = torch.randn_like(sample) * sigma

        # Forward process of diffusion probabilistic model. See https://arxiv.org/pdf/2206.00927.pdf and https://arxiv.org/pdf/2206.00364.pdf
        noisy_sample = sample + noise

        # Condition the noisy sample.
        if self.conditioned_loss:
            noisy_sample = apply_conditions(noisy_sample, conditions, (1 + sigma**2).sqrt())

        # Get scaling factors.
        c_skip, c_out, c_in, c_noise = self.scaling(sigma)

        # Scale and recondition the noisy sample.
        network_input = c_in * noisy_sample
        if self.conditioned_loss:
            network_input = apply_conditions(network_input, conditions, c_in * (1 + sigma**2).sqrt())

        # Prepare extra inputs for the conditional diffusion.
        if extra_inputs is None:
            extra_inputs = TensorDict.from_dict({"global_condition": {}, "local_condition": {}})
        extra_inputs_conditional = extra_inputs.copy()

        # Compute the values of the guides.
        with torch.no_grad():
            values = torch.zeros((noisy_sample.shape[0]), device=noisy_sample.device)
            for name, guide in self.guides.items():
                assert isinstance(guide, BaseGuide), "All guides must be instances of BaseGuide."
                # Get the values of the guides.
                values += guide.get_sample_value(network_input, c_noise, extra_inputs, batch).squeeze(1)

        # Sample the number of zero temperature samples from a binomial distribution.
        n_zero_temperature_samples = np.random.binomial(n=sample.shape[0], p=self.unbiased_loss_probability)

        # Sample the tempature for the biased loss part.
        if self.discrete_temperature_distribution:
            temperature = torch.from_numpy(
                np.random.choice([self.max_temperature, 0, -self.max_temperature], size=(1)),
            ).to(noisy_sample.device, noisy_sample.dtype)
        else:
            temperature = self.max_temperature * (
                2 * torch.rand((1), device=noisy_sample.device, dtype=noisy_sample.dtype) - 1
            )

        # Upsample to the batch size and set zero temperature.
        temperature = torch.repeat_interleave(temperature, noisy_sample.shape[0], dim=0)
        if n_zero_temperature_samples > 0:
            temperature[:n_zero_temperature_samples] = self.temperature_base

        # Set the temperature for the biased loss part.
        extra_inputs_conditional["global_condition"]["temperature"] = (
            temperature.view(noisy_sample.shape[0], 1) / self.max_temperature
        )

        # Compute sample weights for the biased loss part.
        sample_weights = torch.exp(values * temperature - temperature**2 / 2.0)

        # Set unbiased loss part weights to 1.
        if n_zero_temperature_samples > 0:
            sample_weights[:n_zero_temperature_samples] = 1.0

        # Batch normalize the weights per subbatch.
        if self.batch_normalize_weights:
            sample_weights[n_zero_temperature_samples:] /= sample_weights[n_zero_temperature_samples:].mean()

        # Reshape to (batch, 1, 1).
        sample_weights = sample_weights.view([noisy_sample.shape[0]] + [1] * (noisy_sample.dim() - 1))  # (batch, 1, 1)

        # Predict the denoised sample.
        denoised_sample = self.network(network_input, c_noise, concat_extra_inputs(extra_inputs_conditional))
        denoised_sample = c_skip * noisy_sample + c_out * denoised_sample

        # Apply conditioning to the denoised sample.
        if self.mask_conditioned_loss:
            denoised_sample = apply_conditions(denoised_sample, conditions)

        # Compute the denoising error.
        diffusion_loss, diffusion_info = self.loss_fn(denoised_sample, sample, sample_weights=sample_weights / c_out**2)
        loss += diffusion_loss
        infos["losses"]["diffusion"] = diffusion_loss.detach()
        infos["losses_info"]["diffusion"] = diffusion_info

        # Compute the losses of all classifiers.
        for name, guide in self.guides.items():
            assert isinstance(guide, BaseGuide), "All guides must be instances of BaseGuide."
            if guide.is_trained():
                guide_loss, guide_info = guide.loss(
                    noisy_sample_scaled_conditioned=network_input,
                    original_sample_scaled=c_in * sample,
                    c_noise=c_noise,
                    extra_inputs=extra_inputs,
                    noisy_sample_unscaled=noisy_sample,
                    denoised_sample_unscaled=denoised_sample,
                    sigma_unscaled=sigma,
                    batch=batch,
                    denoiser_fn=lambda n, s, e, t: self.forward(n, s, conditions, e, t)[0],
                )
                loss += guide_loss
                infos["losses"][f"classifier_{name}"] = guide_loss.detach()
                infos["losses_info"][f"classifier_{name}"] = guide_info

        return loss, infos
