"""Unguided Diffusion Model."""

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


class UnguidedDiffusion(BaseDiffusionModel):
    """Unguided Diffusion Model."""

    def __init__(
        self,
        network: BaseDiffusionNet,
        guides: Union[DictConfig, torch.nn.ModuleDict],
        # Scaling params.
        scaling: Scaling = EpsilonScaling(),
        # Loss params.
        loss_fn: DiffusionLoss = DiffusionL2(),
        reweight_samples: bool = False,
        reweight_factor: float = 1.0,
        reweight_bias: float = 0.0,
        reweight_key: Optional[str] = None,
        conditioned_loss: bool = True,
        mask_conditioned_loss: bool = False,
    ):
        """Initialize the Unguided Diffusion Model.

        Args:
            network (BaseDiffusionNet): Diffusion network.
            guides (Union[DictConfig, torch.nn.ModuleDict]): Guides. These can be used for training and reweighting the
                batch samples. If a DictConfig, it will be converted to a ModuleDict.
            scaling (Scaling): Scaling object for the diffusion model.
            loss_fn (DiffusionLoss): Loss function for the diffusion model.
            reweight_samples (bool): Whether to reweight the samples based on the key. w = exp(factor*(value-bias))
            reweight_factor (float): Factor by which to reweight the samples.
            reweight_bias (float): Bias to add to the reweighting.
            reweight_key (Optional[str]): Key of the guide to use for reweighting the samples.
            conditioned_loss (bool): Whether to condition the sample for loss computation.
            mask_conditioned_loss (bool): Whether to mask out the conditioned parts in the loss calculation.

        """
        super().__init__()

        # Diffusion and guide Network.
        self.network = network
        if guides is None:
            guides = {}
        self.guides = guides if isinstance(guides, torch.nn.ModuleDict) else torch.nn.ModuleDict(guides)

        # Scaling/preconditioning.
        self.scaling = scaling

        # Loss function parameters.
        self.loss_fn = loss_fn
        self.reweight_samples = reweight_samples
        self.reweight_factor = reweight_factor
        self.reweight_bias = reweight_bias
        self.reweight_key = reweight_key
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
        """Apply one step of diffusion to the noisy sample.

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
                - info: Optional TensorDict containing additional information.

        """
        # Dictionary of additional infos.
        if return_info:
            info = TensorDict({})
        else:
            info = None

        # Compute the denoised sample.
        diffusion_targets = self._compute_diffusion_targets(noisy_sample, sigma, conditions, extra_inputs)
        denoised_sample = diffusion_targets["diffusion"]

        # Apply conditioning to the denoised sample.
        denoised_sample = apply_conditions(denoised_sample, conditions)

        # Add the diffusion target to the info dictionary.
        if info is not None and return_info:
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

        return TensorDict({"diffusion": denoised_sample})

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

        # We can reweight the samples based on the guide values.
        if (
            self.reweight_samples
            and len(self.guides) > 0
            and (self.reweight_key is None or self.reweight_key in self.guides)
        ):
            with torch.no_grad():
                if self.reweight_key is not None:
                    values = self.guides[self.reweight_key].get_sample_value(
                        network_input, c_noise, extra_inputs, batch
                    )
                else:
                    values = torch.zeros((noisy_sample.shape[0], 1), device=noisy_sample.device)
                    for name, guide in self.guides.items():
                        assert isinstance(guide, BaseGuide), "All guides must be instances of BaseGuide."
                        # Get the values of the guides.
                        values += guide.get_sample_value(network_input, c_noise, extra_inputs, batch)
            sample_weights = torch.exp(self.reweight_factor * (values - self.reweight_bias)).view(
                [noisy_sample.shape[0]] + [1] * (noisy_sample.dim() - 1)
            )
            sample_weights = sample_weights / sample_weights.mean(dim=0, keepdim=True)  # Normalize weights.
        else:
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
