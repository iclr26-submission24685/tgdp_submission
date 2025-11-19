"""U-Net with Film embeddings.

Adapted from Scheikl et. al. - Movement Primitive Diffusion
https://github.com/ScheiklP/movement-primitive-diffusion/blob/main/movement_primitive_diffusion/models/conditional_unet1d_inner_model.py
"""

from typing import List, Optional

import torch
from tensordict import TensorDict

from ..embeddings import BaseEmbedding, SinusoidalEmbedding
from ..utils.conv1d import ConditionalResidualBlock1D, Conv1dBlock, Downsample1d, Upsample1d
from .base_diffusion_net import BaseDiffusionNet


class TemporalUNetFilm(BaseDiffusionNet):
    """Temporal UNet model for diffusion models."""

    def __init__(
        self,
        input_size: int,
        sigma_embedding: BaseEmbedding = SinusoidalEmbedding(embedding_size=64),
        global_condition_embedding: Optional[BaseEmbedding] = None,
        local_condition_size: Optional[int] = None,
        down_sizes: List[int] = [512, 1024, 2048],
        kernel_size: int = 5,
        n_groups: int = 8,
        predict_scale_of_condition: bool = True,
    ):
        """Initialize the U-Net model for diffusion.

        Args:
            input_size (int): The size of the input.
            sigma_embedding (BaseEmbedding): The sigma embedding.
            global_condition_embedding (Optional[BaseEmbedding]): The embedding module for the global condition.
            local_condition_size (int, optional): The size of the local condition. One value per timestep and sample.
                This condition is encoded and then added to the sample in the highest down module. Defaults to None.
            down_sizes (List[int], optional): List of sizes for the downsampling layers. Defaults to [512, 1024, 2048].
            kernel_size (int, optional): The size of the kernel for convolutional layers. Defaults to 5.
            n_groups (int, optional): The number of groups for group normalization. Defaults to 8.
            predict_scale_of_condition (bool, optional): Whether to predict the scale of the condition.
                Defaults to True.

        """
        super().__init__()
        all_dims = [input_size] + list(down_sizes)
        start_dim = down_sizes[0]

        # Embed sigma and global conditions from extra_inputs.
        self.sigma_embedding = sigma_embedding
        sigma_embedding_size = self.sigma_embedding.embedding_size
        condition_size = sigma_embedding_size
        if global_condition_embedding is not None:
            self.global_condition_embedding = global_condition_embedding
            condition_size += self.global_condition_embedding.embedding_size

        # Define the input and output sizes for the layers
        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        # Define the local condition embedding. Local conditions are used to condition the model on local features.
        # The local conditions are added to the first down encoder.
        local_condition_embedding = None
        if local_condition_size is not None:
            _, output_channels = in_out[0]
            input_channels = local_condition_size
            local_condition_embedding = torch.nn.ModuleList(
                [
                    # local condition encoder for down encoder
                    ConditionalResidualBlock1D(
                        input_channels,
                        output_channels,
                        condition_size=condition_size,
                        kernel_size=kernel_size,
                        n_groups=n_groups,
                        predict_scale_of_condition=predict_scale_of_condition,
                    ),
                    # local condition encoder for up encoder
                    ConditionalResidualBlock1D(
                        input_channels,
                        output_channels,
                        condition_size=condition_size,
                        kernel_size=kernel_size,
                        n_groups=n_groups,
                        predict_scale_of_condition=predict_scale_of_condition,
                    ),
                ]
            )

        # Define the down modules of the U-Net as two residual blocks followed by a downsampling layer
        # for every dimension in self.down_sizes.
        down_modules = torch.nn.ModuleList([])
        for ind, (input_channels, output_channels) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(
                torch.nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            input_channels,
                            output_channels,
                            condition_size=condition_size,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            predict_scale_of_condition=predict_scale_of_condition,
                        ),
                        ConditionalResidualBlock1D(
                            output_channels,
                            output_channels,
                            condition_size=condition_size,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            predict_scale_of_condition=predict_scale_of_condition,
                        ),
                        Downsample1d(output_channels) if not is_last else torch.nn.Identity(),
                    ]
                )
            )

        # Define the mid modules of the U-Net as two residual blocks for the latent size.
        latent_size = all_dims[-1]
        self.mid_modules = torch.nn.ModuleList(
            [
                ConditionalResidualBlock1D(
                    latent_size,
                    latent_size,
                    condition_size=condition_size,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    predict_scale_of_condition=predict_scale_of_condition,
                ),
                ConditionalResidualBlock1D(
                    latent_size,
                    latent_size,
                    condition_size=condition_size,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    predict_scale_of_condition=predict_scale_of_condition,
                ),
            ]
        )

        # Define the up modules of the U-Net as two residual blocks followed by an upsampling layer
        # for every dimension in self.down_sizes.
        up_modules = torch.nn.ModuleList([])
        for ind, (input_channels, output_channels) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(
                torch.nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            output_channels * 2,
                            input_channels,
                            condition_size=condition_size,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            predict_scale_of_condition=predict_scale_of_condition,
                        ),
                        ConditionalResidualBlock1D(
                            input_channels,
                            input_channels,
                            condition_size=condition_size,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            predict_scale_of_condition=predict_scale_of_condition,
                        ),
                        Upsample1d(input_channels) if not is_last else torch.nn.Identity(),
                    ]
                )
            )

        # Define the final convolutional layer.
        final_conv = torch.nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            torch.nn.Conv1d(start_dim, input_size, 1),
        )

        self.local_condition_embedding = local_condition_embedding
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

    def forward(
        self,
        noisy_sample: torch.Tensor,
        sigma: torch.Tensor,
        extra_inputs: Optional[TensorDict] = None,
    ) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            noisy_sample (torch.Tensor): Noised sample as input to the model. (batch_size, horizon, features)
            sigma (torch.Tensor): Sigma as condition for the model. (batch_size, 1, 1) or ()
            extra_inputs (tensordict.TensorDict, optional): Extra inputs for the model. Defaults to None.
                - TensorDict with key "local_condition" and tensor value (batch_size, horizon, local_condition_size]).
                - TensorDict with key "global_condition" and tensor value (batch_size, global_condition_size).

        Returns:
            torch.Tensor: Model output. (batch_size, horizon, features)

        """
        # Check if horizon and the number of down modules are compatible.
        most_compressed_sample_length = noisy_sample.shape[1] / 2 ** (len(self.down_modules) - 1)
        assert most_compressed_sample_length == int(most_compressed_sample_length), (
            "Sample length must be compatible with the number of down modules."
        )

        # swap from (batch, horizon, features) to (batch, features, horizon) to convolve over time, not features
        noisy_sample = noisy_sample.swapaxes(-1, -2)

        # embed sigma
        if sigma.shape == torch.Size([]):
            sigma = sigma.expand(noisy_sample.shape[0], 1, 1)
        global_feature = self.sigma_embedding(sigma.squeeze(-1))  # (batch, sigma_embedding_size)

        # global features
        if (
            extra_inputs is not None
            and "global_condition" in extra_inputs
            and extra_inputs["global_condition"] is not None
        ):
            global_condition = self.global_condition_embedding(extra_inputs["global_condition"])
            global_feature = torch.cat([global_feature, global_condition], dim=-1)

        # encode local features
        h_local = list()
        if (
            extra_inputs is not None
            and "local_condition" in extra_inputs
            and extra_inputs["local_condition"] is not None
            and self.local_condition_embedding is not None
        ):
            local_condition = extra_inputs["local_condition"]
            local_condition = local_condition.swapaxes(-1, -2)
            resnet, resnet2 = self.local_condition_embedding
            h_local = list([resnet(local_condition, global_feature), resnet2(local_condition, global_feature)])

        # down modules
        x = noisy_sample
        h = []  # list of signals used as skip connections
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):  # type: ignore
            x = resnet(x, global_feature)
            if idx == 0 and len(h_local) > 0:
                x = x + h_local[0]
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        # mid modules
        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        # up modules
        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):  # type: ignore
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            if idx == len(self.up_modules) and len(h_local) > 0:
                x = x + h_local[1]
            x = resnet2(x, global_feature)
            x = upsample(x)

        # final conv
        x = self.final_conv(x)

        # swap back to (batch, horizon, features)
        denoised_sample = x.swapaxes(-1, -2)

        return denoised_sample
