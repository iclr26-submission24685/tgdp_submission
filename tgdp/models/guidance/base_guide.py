"""Base class for guidance models.

Guidance models are supposed to implement both classifier and classifier-free guidance modes.
"""

import logging
from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple

import torch
from tensordict import TensorDict

from ...networks.guides.base_guide_net import BaseGuideNet
from ...utils.extra_inputs import concat_extra_inputs
from ..losses import BaseLoss

logger = logging.getLogger(__name__)


class BaseGuide(torch.nn.Module, ABC):
    """Base class for guidance models.

    Guidance models are supposed to implement both classifer and classifier-free guidance modes.

    A subclass needs to implement the following methods:
    - `get_sample_value`: Compute the value of the given sample during training. This may simply use the information in
        the batch (e.g., when the value is the reward) or use the network (e.g., when the value is a data density). This
        is used during training. It may be called from the loss function of the guide or from the diffusion model (e.g.,
        for weight computation in temperature-guided diffusion).
    - `_loss`: Compute the loss of the model with the current mini-batch. The noise level may be None if the model is
        trained in noise-free mode. The 'loss' method calls this method with the appropriate noise.

    Optionally, a subclass may override the following methods:
    - `_predict_value`: Predicts the value based on the input sample and noise level. The noise level may be None if the
        model is trained in noise-free mode. The 'predict_value' method calls this method with the appropriate noise.
        Per default, this method returns the output of the network. If needed, additional processing steps such as
        summing over the features or applying a non-linearity can be added here. This is used for example to get the
        value of the classifier for example when doing monte-carlo with selection.
    - `get_target_value`: Returns the target value for the guidance model to condition on at inference time. This should
        return a static target value for the guidance model. Per default this returns 1. This should not use the
        network, but may use statistics. These values are used as targets for classifier-free guidance.
    - `gradients`: Returns the gradients of the guidance model with respect to the input sample. For this, we need to
        use the (noise-conditional) network and backprop through it. Per default, this returns the gradients wrt. the
        output of the _predict_value method. If needed, additional processing steps such as masking part of the
        gradients may be added here. This is used for classifier guidance.


    Do not override the 'loss', 'predict_value', or 'is_trained' methods.
    """

    def __init__(
        self,
        train_noise_conditional_network: bool,
        network: Optional[BaseGuideNet] = None,
        loss_fn: Optional[BaseLoss] = None,
    ):
        """Initialize the base guidance model.

        Args:
            train_noise_conditional_network (bool): Flag to indicate whether to train noise-conditional networks
                (for CG) or not (for CFG). The network needs to support the chosen mode.
            network (Optional[BaseClassifierNet]): The classifier network to use for guidance.
            loss_fn (Optional[BaseLoss]): The loss function to use for guidance.

        """
        assert (network is None and loss_fn is None) or (network is not None and loss_fn is not None), (
            "Either both network and loss_fn must be provided, or neither."
        )
        super().__init__()

        # The network and loss function to be used for guidance.
        self.network = network
        self.loss_fn = loss_fn

        # Flag to indicate whether to train noise-conditional networks (for CG) or not (for CFG).
        self.train_noise_conditional_network = train_noise_conditional_network

    ###### Used at inference time ######
    def predict_value(
        self,
        noisy_sample: torch.Tensor,
        sigma: Optional[torch.Tensor],
        extra_inputs: Optional[TensorDict] = None,
    ) -> torch.Tensor:
        """Predict value for the guidance model.

        Predicts the value using the network based on the input sample and noise level. Depending on the training mode,
        this calls '_predict_value' with the appropriate noise level or None. This is used for example to get the value
        of the classifier for example when doing monte-carlo with selection.

        Args:
            noisy_sample (torch.Tensor): The noisy input sample. If the classifier is not noise-conditional, this should
                be noise-free.
            sigma (Optional[torch.Tensor]): The noise level.
            extra_inputs (Optional[TensorDict]): Additional inputs for the model. Defaults to None.

        Returns:
            torch.Tensor: The predicted value based on the input sample and noise level.

        """
        return self._predict_value(noisy_sample, sigma if self.train_noise_conditional_network else None, extra_inputs)

    def _predict_value(
        self,
        noisy_sample: torch.Tensor,
        sigma: Optional[torch.Tensor],
        extra_inputs: Optional[TensorDict],
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
        assert self.network is not None, (
            "You called the predict_value function of a guide that is not trained. This should not happen."
        )
        return self.network(noisy_sample, sigma, concat_extra_inputs(extra_inputs))

    def get_target_value(
        self,
        noisy_sample: torch.Tensor,
        sigma: Optional[torch.Tensor],
        extra_inputs: Optional[TensorDict] = None,
    ) -> torch.Tensor:
        """Return the target value for conditioning is CFG.

        Returns the target value for the guidance model to condition on at inference time. This should
        return a static target value for the guidance model. Per default this returns 1. If the guide has a
        'cfg_target_value', it uses this as target. This should not use the network, but may use statistics. These
        values are used as targets for classifier-free guidance.

        Args:
            noisy_sample (torch.Tensor): The noisy input sample. (batch_size, time, features)
            sigma (Optional[torch.Tensor]): The noise level. (batch_size, 1, 1)
            extra_inputs (Optional[TensorDict]): Additional inputs for the model. Defaults to None.

        Returns:
            torch.Tensor: The target value tensor. (batch_size, 1)

        """
        if not hasattr(self, "cfg_target_value"):
            logger.warning("Trying to get target value from a guide without cfg_target_value. Returning 1.0.")

        return getattr(self, "cfg_target_value", 1.0) * torch.ones(
            (noisy_sample.shape[0], 1), device=noisy_sample.device
        )

    def gradients(
        self,
        noisy_sample: torch.Tensor,
        sigma: torch.Tensor,
        extra_inputs: Optional[TensorDict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the gradients of the guidance model with respect to the input sample.

        Returns the gradients of the guidance model with respect to the input sample. For this, we need to
        use the (noise-conditional) network and backprop through it. Per default, this returns the gradients wrt. the
        output of the _predict_value method. If needed, additional processing steps such as masking part of the
        gradients may be added here. This is used for classifier guidance.

        Args:
            noisy_sample (torch.Tensor): Noised sample as input to the model. (batch_size, time, features)
            sigma (torch.Tensor): Sigma as condition for the model. (batch_size, 1, 1) or ()
            extra_inputs (Optional[TensorDict]): Additional inputs for the model. Defaults to None.

        Returns:
            Tuple(torch.Tensor, torch.Tensor): A tuple containing:
                - The gradients of the guidance output with respect to the input sample.
                - The predicted value(s) from the guidance model for the given input.

        """
        assert self.network is not None, "You are trying to call gradients on a guide without a model."
        assert self.train_noise_conditional_network, (
            "You are trying to call gradients on a guide that is not trained in noise-conditional mode. "
            "This should not happen."
        )

        # Compute gradients w.r.t. the noisy sample.
        noisy_sample.requires_grad_()
        y = self.predict_value(noisy_sample, sigma, extra_inputs)
        grad = torch.autograd.grad([y.sum()], [noisy_sample])[0]
        return y.squeeze(1), grad

    ###### Used at training time ######
    def is_trained(self) -> bool:
        """Check if the guide should be trained."""
        return self.network is not None

    @abstractmethod
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
            torch.Tensor: The value of the given sample during training. (batch_size, 1)

        """
        raise NotImplementedError

    def loss(
        self,
        # Network inputs for value prediction.
        noisy_sample_scaled_conditioned: torch.Tensor,
        original_sample_scaled: torch.Tensor,
        c_noise: torch.Tensor,
        extra_inputs: Optional[TensorDict],
        # Additional inputs to compute the target value.
        noisy_sample_unscaled: torch.Tensor,
        denoised_sample_unscaled: torch.Tensor,
        sigma_unscaled: torch.Tensor,
        batch: TensorDict,
        denoiser_fn: Callable[[torch.Tensor, torch.Tensor, Optional[TensorDict], int], torch.Tensor],
    ) -> Tuple[torch.Tensor, TensorDict]:
        """Compute the loss of the model with the current mini-batch.

        This computes the loss by calling the '_loss' function providing the c_noise or None, depending on the training
        mode. If we train a noise-conditional network, we pass the c_noise, otherwise we pass None.

        Args:
            noisy_sample_scaled_conditioned (torch.Tensor): The noisy sample used as input to the network when we train
                a noise-conditional network. This is scaled and conditioned. (batch_size, time, features)
            original_sample_scaled (torch.Tensor): The original scaled sample used as input to the network when we train
            a noise-free network. (batch_size, time, features)
            c_noise (torch.Tensor): The scaled noise_level. (batch_size, 1, 1) or (1)
            extra_inputs (TensorDict): Additional inputs for the model.
            noisy_sample_unscaled (torch.Tensor): The noisy sample tensor in unscaled space.
            denoised_sample_unscaled (torch.Tensor): The denoised sample tensor in unscaled space as predicted by the
                diffusion model.
            sigma_unscaled (torch.Tensor): The noise level tensor in unscaled space.
            batch (TensorDict): TensorDict containing the current mini-batch.
            denoiser_fn (Callable): The denoiser function, accepting inputs and outputs in unscaled space.
            scale_fn (Callable): The scaling function to be used for the model.
            train_step (int): The current training step.

        Returns:
            Tuple(torch.Tensor, TensorDict): Tuple containing:
            - Loss tensor (1)
            - Additional information about the loss.

        """
        assert self.network is not None, "You called the loss function of a guide that is not trained."

        # If we use noise-conditional networks, we call the network with the noisy sample and sigma.
        # If not, we call it with the original sample and sigma = None.
        return self._loss(
            # Network inputs for value prediction.
            network_input=noisy_sample_scaled_conditioned
            if self.train_noise_conditional_network
            else original_sample_scaled,
            c_noise=c_noise if self.train_noise_conditional_network else None,
            extra_inputs=extra_inputs,
            # Additional inputs to compute the target value.
            noisy_sample_unscaled=noisy_sample_unscaled,
            denoised_sample_unscaled=denoised_sample_unscaled,
            sigma_unscaled=sigma_unscaled,
            batch=batch,
            denoiser_fn=denoiser_fn,
        )

    @abstractmethod
    def _loss(
        self,
        # Network inputs for value prediction.
        network_input: torch.Tensor,
        c_noise: Optional[torch.Tensor],
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
            c_noise (Optional[torch.Tensor]): If we use noise-conditional networks, this is the noise level.
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
        raise NotImplementedError
