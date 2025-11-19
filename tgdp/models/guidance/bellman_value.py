"""Bellman Value Guidance Model.

This module implements a Value guide that is trained using the Bellman equation.
"""

from copy import deepcopy
from typing import Callable, Optional, Tuple

import torch
from tensordict import TensorDict
from tgdp.networks.ema_wrapper import EMANetworkWrapper
from tgdp.utils.extra_inputs import concat_extra_inputs

from ...networks.guides.base_guide_net import BaseGuideNet
from ..losses import BaseLoss
from .base_guide import BaseGuide


class BellmanValueGuide(BaseGuide):
    """Bellman Value Guidance Model.

    BellmanValueGuide provides guidance for the diffusion process by predicting the normalized discounted return
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
        discount: float = 0.997,
        target_network_decay: float = 0.995,
    ):
        """Initialize the bellman value guidance model.

        Args:
            train_noise_conditional_network (bool): Flag to indicate whether to use classifier gradients.
            network (Optional[BaseClassifierNet]): The classifier network to be used if classifier guidance is enabled.
            loss_fn (Optional[BaseLoss]): The loss function to be used if classifier guidance is enabled.
            cfg_target_value (float): The target value to be used if classifier guidance is disabled.
            discount (float): Discount factor for future rewards in the Bellman equation.
            target_network_decay (float): The decay factor for the target network ema update.

        """
        super().__init__(train_noise_conditional_network, network, loss_fn)
        assert self.network is not None, "Bellman Value guides need a network."
        self.network: BaseGuideNet

        # Target and trained network.
        if isinstance(self.network, EMANetworkWrapper):
            self.target_network = deepcopy(self.network._original).requires_grad_(False)
        else:
            self.target_network = deepcopy(self.network).requires_grad_(False)

        # Discount factor and target network decay used in the loss function.
        self.target_network_decay = target_network_decay
        self.discount = discount

        # If we use classifier-free guidance, we need to set the target value.
        self.cfg_target_value = cfg_target_value

    def _predict_value(
        self, noisy_sample: torch.Tensor, sigma: Optional[torch.Tensor], extra_inputs: Optional[TensorDict]
    ) -> torch.Tensor:
        """Predict value for the guidance model.

        Predicts the value based on the input sample and noise level. The noise level may be None if the
        model is trained in noise-free mode. The 'predict_value' method calls this method with the appropriate noise.
        Per default, this method returns the output of the network. If needed, additional processing steps such as
        summing over the features or applying a non-linearity can be added here. This is used for example to get the
        value of the classifier for example when doing monte-carlo with selection.

        Args:
            noisy_sample (torch.Tensor): The noisy input sample.
            sigma (Optional[torch.Tensor]): The noise level. For noise-free guidance, this is None.
            extra_inputs (TensorDict): Additional inputs for the model. Defaults to None.

        Returns:
            torch.Tensor: The predicted value based on the input sample and noise level.

        """
        return super()._predict_value(noisy_sample[:, 1:], sigma, extra_inputs).sum(dim=1)

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
        # Compute the target values. We predict the value of all states and sum over it.
        return self.network(
            noisy_sample,
            sigma if self.train_noise_conditional_network else None,
            concat_extra_inputs(extra_inputs),
        ).sum(dim=1)

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

        # Extract values from batch.
        # data_lengths_m1 = batch["padding"].shape[1] - batch["padding"].sum(dim=1) - 1
        # arange = torch.arange(batch["padding"].shape[0]).to(data_lengths_m1)
        # idxs = torch.floor(torch.rand_like(data_lengths_m1, dtype=torch.float) * data_lengths_m1).to(torch.long)
        # rewards = batch["rewards"][arange, idxs].unsqueeze(-1)
        # terminals = batch["terminations"][arange, idxs + 1].float().unsqueeze(-1)
        # obs_t = network_input[arange, idxs]
        # obs_tp1 = network_input[arange, idxs + 1]

        # # Compute the target value using the Bellman equation and the target network.
        # next_values = self.target_network(obs_tp1, c_noise, concat_extra_inputs(extra_inputs))
        # target_values = rewards + self.discount * next_values * (1 - terminals)
        # target_values = target_values.detach()

        # # Predict the values using the trained network.
        # current_values = self.network(obs_t, c_noise, concat_extra_inputs(extra_inputs))
        # current_values = current_values

        # Extract values from batch.
        not_padding = 1 - batch["padding"][:, 1:].float().unsqueeze(-1)
        rewards = batch["rewards"][:, :-1].unsqueeze(-1)
        terminals = batch["terminations"][:, 1:].float().unsqueeze(-1)
        obs_t = network_input[:, :-1]
        obs_tp1 = network_input[:, 1:]

        # Compute the target value using the Bellman equation and the target network.
        next_values = self.target_network(obs_tp1, c_noise, concat_extra_inputs(extra_inputs))
        target_values = rewards + self.discount * next_values * (1 - terminals)
        target_values = (target_values * not_padding).detach()

        # Predict the values using the trained network.
        current_values = self.network(obs_t, c_noise, concat_extra_inputs(extra_inputs))
        current_values = current_values * not_padding

        # Compute the loss
        loss, info = self.loss_fn(pred=current_values, targ=target_values)

        # Update the target_network.
        for param, target_param in zip(self.network.parameters(), self.target_network.parameters()):
            target_param.data.copy_(
                target_param.data * self.target_network_decay + param.data * (1 - self.target_network_decay)
            )

        return loss, info
