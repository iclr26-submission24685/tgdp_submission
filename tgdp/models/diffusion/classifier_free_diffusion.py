"""Classifier-Free Diffusion Model."""

import logging
from typing import Optional, Tuple, Union

import torch
from omegaconf.dictconfig import DictConfig
from tensordict import TensorDict

from ...inpainting_conditioning import ConditionType, apply_conditions
from ...networks.diffusion.base_diffusion_net import BaseDiffusionNet
from ...utils.extra_inputs import concat_extra_inputs
from ..guidance.base_guide import BaseGuide
from ..losses import DiffusionL2, DiffusionLoss
from ..scaling import EpsilonScaling, Scaling
from .base_diffusion import BaseDiffusionModel

logger = logging.getLogger(__name__)


class ClassifierFreeDiffusion(BaseDiffusionModel):
    """Classifier-Free Diffusion Model."""

    def __init__(
        self,
        network: BaseDiffusionNet,
        guides: Union[DictConfig, torch.nn.ModuleDict],
        guide_scale: float,
        dropout_probability: float,
        empty_set_value: float = 0.0,
        scaling: Scaling = EpsilonScaling(),
        loss_fn: DiffusionLoss = DiffusionL2(),
        conditioned_loss: bool = True,
        mask_conditioned_loss: bool = False,
    ):
        """Initialize the Classifier-Free Diffusion Model.

        Args:
            network: Diffusion network.
            guides: Dictionary containing the classifier guidance networks.
            guide_scale: The scale of the conditioned diffusion target. Should be in [0,1].
            dropout_probability: The probability of dropping a condition. If None, the probability is set to 0.0.
            empty_set_value: Value to use for empty set conditions in the guides.
            scaling: Scaling object for the diffusion model.
            loss_fn: Loss function for the diffusion model.
            conditioned_loss: Whether to condition the sample for loss computation.
            mask_conditioned_loss: Whether to mask out the conditioned parts in the loss calculation.

        """
        assert 0 <= dropout_probability <= 1, "Dropout probability must be in [0,1]."
        super().__init__()

        # Diffusion Network and Guides.
        self.network = network
        if guides is None:
            guides = {}
        self.guides = guides if isinstance(guides, torch.nn.ModuleDict) else torch.nn.ModuleDict(guides)

        # Guide scale and dropout probability.
        self.guide_scale = guide_scale
        self.dropout_probability = dropout_probability
        self.empty_set_value = empty_set_value

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
            diffusion_step: The diffusion step from 0 to T-1.
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
        diffusion_targets = self._compute_diffusion_targets(noisy_sample, sigma, conditions, extra_inputs)

        # Apply the weighting of diffusion targets.
        denoised_sample = (1 - self.guide_scale) * diffusion_targets[
            "unconditional"
        ] + self.guide_scale * diffusion_targets["conditional"]

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
    ) -> TensorDict:
        """Compute the diffusion target for the noisy sample.

        Args:
            noisy_sample: Noisy sample tensor. (batch_size, time, features)
            sigma: Noise level tensor. (batch_size, 1, 1) or ()
            conditions: Dictionary containing the conditions.
            extra_inputs: Extra inputs dictionary.

        Returns:
            Diffusion targets dictionary for all diffusion models (conditional and unconditional).
                {key: (batch_size, time, features)}

        """
        # Get scaling factors.
        c_skip, c_out, c_in, c_noise = self.scaling(sigma)

        # Scale and condition the noisy sample.
        noisy_sample = apply_conditions(noisy_sample, conditions, (1 + sigma**2).sqrt())
        network_input = c_in * noisy_sample

        # Compile the unconditional and conditional extra inputs.
        if extra_inputs is None:
            extra_inputs = TensorDict.from_dict({"global_condition": {}, "local_condition": {}})
        extra_inputs_conditional = extra_inputs.copy()
        extra_inputs_unconditional = extra_inputs.copy()
        for name, guide in self.guides.items():
            assert isinstance(guide, BaseGuide), f"Guide {name} is not a BaseGuide instance."
            conditional_values = guide.get_target_value(network_input, c_noise, extra_inputs)
            extra_inputs_conditional["global_condition"][f"{name}"] = conditional_values
            extra_inputs_unconditional["global_condition"][f"{name}"] = self.empty_set_value * torch.ones_like(
                conditional_values
            )

        # Compute the denoising targets of the unconditional diffusion.
        denoised_sample_unconditional = self.network(
            network_input, c_noise, concat_extra_inputs(extra_inputs_unconditional)
        )
        denoised_sample_unconditional = c_skip * noisy_sample + c_out * denoised_sample_unconditional

        # Compute the denoising targets of the conditional diffusion.
        denoised_sample_conditional = self.network(
            network_input, c_noise, concat_extra_inputs(extra_inputs_conditional)
        )
        denoised_sample_conditional = c_skip * noisy_sample + c_out * denoised_sample_conditional

        # Return the diffusion targets.
        diffusion_targets = TensorDict(
            {"unconditional": denoised_sample_unconditional, "conditional": denoised_sample_conditional}
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
            noisy_sample: Noisy sample tensor. If the classifier is not noise-conditional, this should be noise-free.
                (batch_size, time, features)
            sigma: Noise level tensor. (batch_size, 1, 1) or ()
            conditions: Dictionary containing the conditions.
            extra_inputs: Extra inputs dictionary.
            key: Key of the guide to compute the value for. If None, all values are computed.

        Returns:
            values: Tensor containing the value of the classifier for the given key. If key is None a TensorDict
                with all values is returned.

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
            assert isinstance(guide, BaseGuide), f"Guide {key} is not a BaseGuide instance."
            value = guide.predict_value(noisy_sample_scaled_and_conditioned, c_noise, extra_inputs)
            return value
        else:
            values = TensorDict({})
            for guide_name, guide in self.guides.items():
                # Compute the values of the guides.
                assert isinstance(guide, BaseGuide), f"Guide {guide_name} is not a BaseGuide instance."
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
        """Compute the loss of the model with the current mini-batch.

        Args:
            sample (torch.Tensor): The original sample tensor. (batch_size, time, features)
            sigma (torch.Tensor): Noise level tensor. (batch_size, 1, 1) or ()
            conditions (Optional[ConditionType]): Dictionary containing the conditions.
            batch (TensorDict): TensorDict containing the current mini-batch.
            extra_inputs (Optional[TensorDict]): Extra inputs dictionary.

        Returns:
            Tuple(torch.Tensor, TensorDict): The loss value and additional information.
                - loss: The loss value.
                - infos: TensorDict containing additional information.

        """
        # Dictionary of additional infos.
        infos = TensorDict({})
        infos["losses"] = TensorDict({})
        infos["losses_info"] = TensorDict({})

        # Initialize loss.
        loss = 0.0

        # Noise is first drawn from a normal distribution with mean 0 and std 1, then scaled by sigma (the desired std)
        noise = torch.randn_like(sample) * sigma

        # Forward process of diffusion probabilistic model. See https://arxiv.org/pdf/2206.00927.pdf and https://arxiv.org/pdf/2206.00364.pdf
        noisy_sample = sample + noise

        # Condition the noisy sample.
        if self.conditioned_loss:
            noisy_sample = apply_conditions(noisy_sample, conditions, (1 + sigma**2).sqrt())

        # Get scaling factors
        c_skip, c_out, c_in, c_noise = self.scaling(sigma)

        # Scale and recondition the noisy sample.
        network_input = c_in * noisy_sample
        if self.conditioned_loss:
            network_input = apply_conditions(network_input, conditions, c_in * (1 + sigma**2).sqrt())

        # Compile extra inputs depending on whether to use dropout or not.
        if extra_inputs is None:
            extra_inputs = TensorDict.from_dict({"global_condition": {}, "local_condition": {}})
        extra_inputs_conditional = extra_inputs.copy()
        dropout_mask = torch.rand((noisy_sample.shape[0], 1), device=noisy_sample.device) < self.dropout_probability
        with torch.no_grad():
            for name, guide in self.guides.items():
                # Get the values of the guides.
                assert isinstance(guide, BaseGuide), f"Guide {name} is not a BaseGuide instance."
                values = dropout_mask * self.empty_set_value + torch.logical_not(dropout_mask) * guide.get_sample_value(
                    network_input, c_noise, extra_inputs, batch
                )
                extra_inputs_conditional["global_condition"][f"{name}"] = values

        # Predict the denoised sample (depending on the Scaling, the internal network predicts the mean of the denoised
        # sample, the noise, or sth in between).
        denoised_sample = self.network(network_input, c_noise, concat_extra_inputs(extra_inputs_conditional))
        denoised_sample = c_skip * noisy_sample + c_out * denoised_sample

        # Apply conditioning to the denoised sample. This effectively masks the conditioned parts in the loss.
        if self.mask_conditioned_loss:
            denoised_sample = apply_conditions(denoised_sample, conditions)

        sample_weights = torch.ones(
            ([noisy_sample.shape[0]] + [1] * (noisy_sample.dim() - 1)), device=noisy_sample.device
        )

        # Compute the denoising error. See https://arxiv.org/pdf/2206.00364.pdf
        # (Karras et. al. scale samples by 1/c_out**2, many impementations don't)
        sample_weights /= c_out**2
        diffusion_loss, diffusion_info = self.loss_fn(denoised_sample, sample, sample_weights=sample_weights)
        loss += diffusion_loss
        infos["losses"]["diffusion"] = diffusion_loss.detach()
        infos["losses_info"]["diffusion"] = diffusion_info

        # Compute the losses of all classifiers. We pass many arguments to the classifier models. Not all are used
        # by all classifiers.
        for name, guide in self.guides.items():
            assert isinstance(guide, BaseGuide), f"Guide {name} is not a BaseGuide instance."
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
