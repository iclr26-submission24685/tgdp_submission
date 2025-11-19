"""Monte Carlo Return Guidance Model.

This module implements the Monte Carlo Return Guidance model, which is used to guide the diffusion process
by predicting the normalized discounted return starting at the current step. It can operate in two modes:
1. Classifier Guidance (CG): Uses a noise-conditional network to predict the value for each noise level.
2. Classifier-Free Guidance (CFG): Uses a noise-free network to predict the value only once with zero noise.
"""

from typing import Callable, Optional, Tuple

import torch
from tensordict import TensorDict

from ...networks.guides.base_guide_net import BaseGuideNet
from ..losses import BaseLoss
from .base_guide import BaseGuide


class MonteCarloReturnGuide(BaseGuide):
    """Monte Carlo Return Guidance Model.

    MonteCarloReturnGuide provides guidance for the diffusion process by predicting the normalized discounted return
    starting at the current step, supporting both classifier guidance (CG) and classifier-free guidance (CFG) modes.
    """

    def __init__(
        self,
        train_noise_conditional_network: bool,
        # Needed for Classifier Guidance
        network: Optional[BaseGuideNet] = None,
        loss_fn: Optional[BaseLoss] = None,
        # Needed for Classifier-Free Guidance
        cfg_target_value: Optional[float] = None,
    ):
        """Initialize the monte-carlo return guidance model.

        Args:
            train_noise_conditional_network (bool): Flag to indicate whether to use classifier gradients.
            network (Optional[BaseClassifierNet]): The classifier network to be used if classifier guidance is enabled.
            loss_fn (Optional[BaseLoss]): The loss function to be used if classifier guidance is enabled.
            cfg_target_value (float): The target value to be used if classifier guidance is disabled.

        """
        super().__init__(train_noise_conditional_network, network, loss_fn)

        # If we use classifier-free guidance, we need to set the target value.
        self.cfg_target_value = cfg_target_value

    def get_sample_value(
        self,
        noisy_sample: torch.Tensor,
        sigma: Optional[torch.Tensor],
        extra_inputs: Optional[TensorDict],
        batch: TensorDict,
    ) -> torch.Tensor:
        """Compute the value of the given sample during training.

        Compute the value of the given sample during training. This may simply use the information in the batch (e.g.,
        when the value is the reward) or use the network (e.g., when the value is a data density). This is used during
        training. It may be called from the loss function of the guide or from the diffusion model (e.g., for weight
        computation in temperature-guided diffusion).

        Args:
            noisy_sample (torch.Tensor): Noised sample as input to the model. (batch_size, time, features)
            sigma (Optional[torch.Tensor]): Sigma as condition for the model. (batch_size, 1, 1) or ()
            extra_inputs (TensorDict): Additional inputs for the model. Defaults to None.
            batch (TensorDict): TensorDict containing the current mini-batch.

        Returns:
            torch.Tensor: The value of the given sample during training. (batch_size, 1

        """
        # Compute the target values. We try to predict the discounted return starting at the current step.
        return batch["mc_returns"][:, 0].unsqueeze(1)

    def _loss(
        self,
        # Network inputs for value prediction.
        network_input: torch.Tensor,
        c_noise: torch.Tensor | None,
        extra_inputs: Optional[TensorDict],
        # Additional inputs to compute the target value.
        noisy_sample_unscaled: torch.Tensor,
        denoised_sample_unscaled: torch.Tensor,
        sigma_unscaled: torch.Tensor,
        batch: TensorDict,
        denoiser_fn: Callable[[torch.Tensor, torch.Tensor, Optional[TensorDict], int], torch.Tensor],
    ) -> Tuple[torch.Tensor, TensorDict]:
        """Compute the loss of the given sample.

        Compute the loss of the model with the current mini-batch. The noise level may be None if the model is
        trained in noise-free mode. The 'loss' method calls this method with the appropriate noise. Needs to be
        implemented by subclasses.

        Args:
            network_input (torch.Tensor): The input to the network. If we use noise-conditional networks, this is
                the noisy sample. If not, this is the original sample.
            c_noise (torch.Tensor | None): If we use noise-conditional networks, this is the noise level.
                If not, this is None.
            extra_inputs (TensorDict): Additional inputs for the model.
            sample (torch.Tensor): The unscaled sample.
            noisy_sample_unscaled (torch.Tensor): The unscaled noisy sample.
            denoised_sample_unscaled (torch.Tensor): The unscaled denoised sample.
            sigma_unscaled (torch.Tensor): The unscaled noise level.
            batch (TensorDict): Additional inputs for the model.
            denoiser_fn (Callable): Denoiser function for the model.

        Returns:
            Tuple(torch.Tensor, TensorDict): Tuple containing.
            - Loss tensor (1)
            - Additional information about the loss.


        """
        assert self.loss_fn is not None, "You are trying to call loss on a guide without a loss function."
        # Compute the target value
        target_value = self.get_sample_value(network_input, c_noise, extra_inputs, batch)

        # Predict the the value.
        predicted_value = self._predict_value(network_input, c_noise, extra_inputs)  # (batch_size, 1)

        # Compute the loss
        loss, info = self.loss_fn(pred=predicted_value, targ=target_value)

        return loss, info
