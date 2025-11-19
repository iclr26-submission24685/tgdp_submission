"""Classifier-Guided Diffusion Model."""

import logging
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from omegaconf.dictconfig import DictConfig
from tensordict import TensorDict

from ...inpainting_conditioning import ConditionType, apply_conditions
from ...networks.diffusion.base_diffusion_net import BaseDiffusionNet
from ...utils.dict_buffer import DictBuffer
from ...utils.extra_inputs import concat_extra_inputs
from ..guidance.base_guide import BaseGuide
from ..losses import DiffusionL2, DiffusionLoss
from ..scaling import EpsilonScaling, Scaling
from .base_diffusion import BaseDiffusionModel

logger = logging.getLogger(__name__)


class ClassifierGuidedDiffusion(BaseDiffusionModel):
    """Classifier-Guided Diffusion Model."""

    def __init__(
        self,
        network: BaseDiffusionNet,
        guides: Union[DictConfig, torch.nn.ModuleDict],
        # Guidance and sampling params.
        guide_scales: Optional[Dict[str, float]] = None,
        n_guide_steps: int = 2,
        scale_grad_by_var: bool = True,
        clip_gradients: bool = False,
        clip_gradients_value: float = 1.0,
        normalize_gradients: bool = False,
        clip_var_scale_lower: Optional[float] = None,
        clip_var_scale_upper: Optional[float] = None,
        sigma_stop_grad: float = 0.0,
        clip_diffusion_targets_to_noise_level: bool = False,
        # Scaling params.
        scaling: Scaling = EpsilonScaling(),
        # Loss params.
        loss_fn: DiffusionLoss = DiffusionL2(),
        guide_noise_dropout_probability: float = 0.0,
        conditioned_loss: bool = True,
        mask_conditioned_loss: bool = False,
    ):
        """Initialize the Classifier-Guided Diffusion Model.

        Args:
            network (BaseDiffusionNet): Diffusion network.
            guides (Union[DictConfig, torch.nn.ModuleDict]): Dictionary containing the classifier guidance networks.
            guide_scales (Optional[Dict[str, float]]): Dictionary containing the scales of the guides. If None, all
                scales are set to 1.
            n_guide_steps (int): Number of guide steps.
            scale_grad_by_var (bool): Whether to scale the gradients by the noise level.
            clip_gradients (bool): Whether to clip the gradients during guidance. Cannot be used together with
                `normalize_gradients`.
            clip_gradients_value (float): Value to which gradients are clipped if `clip_gradients` is True.
            normalize_gradients (bool): Whether to normalize the gradients during guidance. Cannot be used together with
                `clip_gradients`.
            clip_var_scale_lower (Optional[float]): When scaling gradients by variance, we clip this scaling factor to t
                his lower bound.
            clip_var_scale_upper (Optional[float]): When scaling gradients by variance, we clip this scaling factor to
                this upper bound.
            sigma_stop_grad (float): Noise level at which to stop the classifier-gradients.
            clip_diffusion_targets_to_noise_level (bool): Whether to clip the diffusion targets to the noise level.
            scaling (Scaling): Scaling object for the diffusion model.
            loss_fn (DiffusionLoss): Loss function for the diffusion model.
            guide_noise_dropout_probability (float): Probability of dropout for setting the guide noise to zero. This is
                useful, if we want guides that predicts the noise-free value for MC Selection.
            conditioned_loss (bool): Whether to condition the sample for loss computation.
            mask_conditioned_loss (bool): Whether to mask out the conditioned parts in the loss calculation.

        """
        super().__init__()

        # Diffusion and Guide Networks.
        self.network = network
        if guides is None:
            guides = {}
        self.guides = guides if isinstance(guides, torch.nn.ModuleDict) else torch.nn.ModuleDict(guides)
        for guide in guides.values():
            assert guide.network is not None, "For classifer-guided diffusion, all guides need a network."

        # Guide scales. The DictBuffer adds all values as buffers, since they may be adapted during training.
        if guide_scales is not None:
            self.guide_scales = DictBuffer(guide_scales)
        else:
            self.guide_scales = DictBuffer({k: 1.0 for k in guides.keys()})
        assert np.all([k in self.guide_scales for k in guides.keys()]), "Scales must be provided for all guides."

        # Sampling parameters.
        self.n_guide_steps = n_guide_steps
        self.scale_grad_by_var = scale_grad_by_var
        self.clip_gradients = clip_gradients
        self.clip_gradients_value = clip_gradients_value
        self.normalize_gradients = normalize_gradients
        assert not clip_gradients or not normalize_gradients, (
            "Cannot use clip_gradients and normalize_gradients at the same time. Please choose one of them."
        )
        self.clip_var_scale_lower = clip_var_scale_lower if clip_var_scale_lower is not None else 0.0
        self.clip_var_scale_upper = clip_var_scale_upper if clip_var_scale_upper is not None else np.inf
        self.sigma_stop_grad = sigma_stop_grad
        self.clip_diffusion_targets_to_noise_level = clip_diffusion_targets_to_noise_level

        # Scaling/preconditioning.
        self.scaling = scaling

        # Loss function parameters.
        self.loss_fn = loss_fn
        self.guide_noise_dropout_probability = guide_noise_dropout_probability
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
        """Apply one step of classifier-guided diffusion to the noisy sample.

        Args:
            noisy_sample (torch.Tensor): Noisy sample tensor. (batch_size, time, features)
            sigma (torch.Tensor): Noise level tensor. (batch_size, 1, 1) or ()
            conditions (Optional[ConditionType]): Dictionary containing the conditions.
            extra_inputs (Optional[TensorDict]): Extra inputs dictionary.
            diffusion_step (int): Diffusion step from 0 to T-1.
            return_info (bool): Whether to return additional information.

        Returns:
            Tuple(torch.Tensor, Optional[TensorDict]): Tuple containing:
                - denoised_sample: Denoised sample tensor. (batch_size, time, features)
                - info: Optional TensorDict containing additional information, such as values and gradients of the
                    classifier guidance networks.

        """
        # Dictionary of additional infos.
        if return_info:
            info = TensorDict({})
        else:
            info = None

        # Compute guide gradients. Note that the gradients are scaled by the guide scales and potentially by sigma**2.
        values, classifier_gradients = self._compute_classifier_values_and_gradients(
            noisy_sample, sigma, conditions, extra_inputs
        )

        # Apply the classifier gradients to the noisy sample.
        guided_sample = noisy_sample + sum([v for k, v in classifier_gradients.items()])

        # Compute the denoised sample.
        diffusion_targets = self._compute_diffusion_targets(guided_sample, sigma, conditions, extra_inputs)
        denoised_sample = diffusion_targets["diffusion"]

        # Apply conditioning to the denoised sample.
        denoised_sample = apply_conditions(denoised_sample, conditions)

        # Add values, classifier gradients, and the diffusion target to the info dictionary.
        if info is not None and return_info:
            info["values"] = values
            info["classifier_gradients"] = classifier_gradients
            info["diffusion_targets"] = diffusion_targets

        return denoised_sample, info

    def _compute_diffusion_targets(
        self,
        noisy_sample: torch.Tensor,
        sigma: torch.Tensor,
        conditions: Optional[ConditionType],
        extra_inputs: Optional[TensorDict] = None,
    ) -> TensorDict:
        """Compute the diffusion target for the noisy sample.

        Args:
            noisy_sample: Noisy sample tensor. (batch_size, time, features)
            sigma: Noise level tensor. (batch_size, 1, 1) or ()
            conditions: Dictionary containing the conditions.
            extra_inputs: Extra inputs dictionary.

        Returns:
            denoised_sample: TensorDict of denoised samples. For CG this has just one entry: 'diffusion'.
                (batch_size, time, features)

        """
        # Get scaling factors.
        c_skip, c_out, c_in, c_noise = self.scaling(sigma)

        # Scale and condition the noisy sample.
        noisy_sample = apply_conditions(noisy_sample, conditions, (1 + sigma**2).sqrt())
        network_input = c_in * noisy_sample

        # Compute the denoised sample by first passing the (scaled) noisy sample through the network and then applying
        # the scaling factors (including skip connection).
        denoised_sample = self.network(network_input, c_noise, concat_extra_inputs(extra_inputs))
        denoised_sample = c_skip * noisy_sample + c_out * denoised_sample

        # Clip denoised to noise level.
        if self.clip_diffusion_targets_to_noise_level:
            denoised_sample.clamp_(min=-((1 + sigma**2) ** 0.5), max=(1 + sigma**2) ** 0.5)

        return TensorDict({"diffusion": denoised_sample})

    def _compute_classifier_values_and_gradients(
        self,
        noisy_sample: torch.Tensor,
        sigma: torch.Tensor,
        conditions: Optional[ConditionType],
        extra_inputs: Optional[TensorDict] = None,
    ) -> Tuple[TensorDict, TensorDict]:
        """Compute the values gradients of the classifier guidance networks w.r.t. the input.

        This computes the value and gradients of the classifier guidance networks w.r.t. the input sample.
        Input sample and output gradients are in unscaled space.

        Args:
            noisy_sample: Noisy sample tensor. (batch_size, time, features)
            sigma: Noise level tensor. (batch_size, 1, 1) or ()
            conditions: Dictionary containing the conditions.
            extra_inputs: Extra inputs dictionary.

        Returns:
            gradients: TensorDict containing the gradients of the classifier guidance networks.
            values: TensorDict containing the values of the classifier guidance networks.

        """
        # Dictionary of gradients and values.
        gradients = TensorDict({k: torch.zeros_like(noisy_sample) for k in self.guides.keys()})
        values = TensorDict(
            {
                k: torch.zeros((noisy_sample.shape[0], 1), device=noisy_sample.device, dtype=noisy_sample.dtype)
                for k in self.guides.keys()
            }
        )

        # Get scaling factors
        _, _, c_in, c_noise = self.scaling(sigma)

        if len(self.guides) > 0:
            # Set starting point for guide evaluations.
            guided_sample = noisy_sample

            # Apply classifier guidance for all guides for n_guide_steps.
            for _ in range(self.n_guide_steps):
                # We evaluate all guides at the same starting point.
                guided_sample_start_scaled_and_conditioned = apply_conditions(
                    c_in * guided_sample, conditions, c_in * (1 + sigma**2).sqrt()
                )

                # Iterate over all guides.
                for guide_name, guide in self.guides.items():
                    # Compute the gradients of the guides.
                    with torch.enable_grad():
                        assert isinstance(guide, BaseGuide), "All guides must be instances of BaseGuide."
                        value, grads = guide.gradients(
                            guided_sample_start_scaled_and_conditioned, c_noise, extra_inputs
                        )

                    # Clip or normalize the gradients if necessary.
                    # NOTE We clip/normalize in scaled space. Does this make sense?
                    if self.clip_gradients:
                        grads = torch.clip(grads, -self.clip_gradients_value, self.clip_gradients_value)
                    elif self.normalize_gradients:
                        norm = torch.linalg.vector_norm(grads, dim=(-1, -2), keepdim=True)
                        norm = torch.clip(norm, min=1e-8)  # Avoid division by zero.
                        grads = grads / norm

                    # Scale gradients by c_in to get gradients in unscaled space.
                    grads = c_in * grads

                    # Scale the gradients by the noise level. In https://openreview.net/pdf?id=PxTIG12RRHS, it is not
                    # scaled. Since we scale the input by c_in, the variance of the input is (c_in*sigma)**2.
                    if self.scale_grad_by_var:
                        grads = (
                            torch.clip(torch.square(c_in * sigma), self.clip_var_scale_lower, self.clip_var_scale_upper)
                            * grads
                        )

                    # Scale gradients by the guide's scale and update the sample.
                    grads = self.guide_scales[guide_name] * grads

                    # Stop gradients if sigma is below sigma_stop_grad.
                    if sigma < self.sigma_stop_grad:
                        grads = torch.zeros_like(grads)

                    # Store the gradients and update sample.
                    gradients[guide_name] += grads
                    guided_sample = guided_sample + grads

                    # Store the last value of the guide.
                    values[guide_name] = value

        return values, gradients

    def compute_classifier_values(
        self,
        noisy_sample: torch.Tensor,
        sigma: torch.Tensor,
        conditions: Optional[ConditionType],
        extra_inputs: Optional[TensorDict] = None,
        key: Optional[str] = None,
    ) -> Union[torch.Tensor, TensorDict]:
        """Compute the values of the classifier guidance networks.

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
            assert isinstance(guide, BaseGuide), "The guide must be an instance of BaseGuide."
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
        infos = TensorDict.from_dict({"losses": TensorDict({}), "losses_info": TensorDict({})})

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

        # Predict the denoised sample (depending on the Scaling, the internal network predicts the mean of the denoised
        # sample, the noise, or sth in between).
        denoised_sample = self.network(network_input, c_noise, concat_extra_inputs(extra_inputs))
        denoised_sample = c_skip * noisy_sample + c_out * denoised_sample

        # Apply conditioning to the denoised sample.
        if self.mask_conditioned_loss:
            denoised_sample = apply_conditions(denoised_sample, conditions)

        # Compute the denoising error. See https://arxiv.org/pdf/2206.00364.pdf
        # (Karras et. al. scale samples by 1/c_out**2, many impementations don't)
        diffusion_loss, diffusion_info = self.loss_fn(denoised_sample, sample, sample_weights=1 / c_out**2)
        loss += diffusion_loss
        infos["losses"]["diffusion"] = diffusion_loss.detach()
        infos["losses_info"]["diffusion"] = diffusion_info

        # Compute noise dropout. We use this to train the noise free classifier alongside the noise-conditioned one.
        if self.guide_noise_dropout_probability > 0.0:
            # Apply dropout to the sample and the sigma tensors.
            dropout_mask = (
                torch.rand((noisy_sample.shape[0], 1, 1), device=noisy_sample.device)
                < self.guide_noise_dropout_probability
            )
            dropout_c_in, _, _, dropout_c_noise = self.scaling(torch.zeros_like(sigma))
            network_input = dropout_mask * dropout_c_in * sample + torch.logical_not(dropout_mask) * network_input
            c_noise = dropout_mask * dropout_c_noise + torch.logical_not(dropout_mask) * c_noise

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
