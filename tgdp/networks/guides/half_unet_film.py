"""Half U-Net with FiLM conditioning."""

from typing import List, Optional

import torch
from tensordict import TensorDict
from torch import nn

from ..embeddings import BaseEmbedding
from ..utils.conv1d import ConditionalResidualBlock1D, Conv_and_Sum, Downsample1d
from .base_guide_net import BaseGuideNet


# Adapted from Scheikl et. al. - Movement Primitive Diffusion
# https://github.com/ScheiklP/movement-primitive-diffusion/blob/main/movement_primitive_diffusion/models/conditional_unet1d_inner_model.py
class TemporalHalfUNetFilm(BaseGuideNet):
    """Half of a U-Net for temporal data.

    This model is a temporal U-Net that processes input data with a temporal dimension (e.g., time series data).
    It takes in a tensor and predicts either a single values or one value per timestep. Can be used as a classifier.
    """

    def __init__(
        self,
        input_size: int,
        sigma_embedding: Optional[BaseEmbedding] = None,
        global_condition_embedding: Optional[BaseEmbedding] = None,
        local_condition_size: Optional[int] = None,
        down_sizes: List[int] = [512, 1024, 2048],
        fully_connected_size: int = 512,
        out_size: int = 1,
        kernel_size: int = 5,
        n_groups: int = 8,
        downsample: bool = True,
        final_activation: nn.Module = nn.Identity(),
        predict_scale_of_condition: bool = True,
    ):
        """Initialize the HalfUNet model.

        Args:
            input_size (int): The size of the input.
            sigma_embedding (Optional[BaseEmbedding]): The sigma embedding.
            global_condition_embedding (Optional[BaseEmbedding]): The global condition embedding.
            local_condition_size (int, optional): The size of the local condition. Default is None.
            down_sizes (List[int], optional): List of sizes for the downsampling layers. Default is [512, 1024, 2048].
            fully_connected_size (int, optional): Dimension of the fully connected layer. Default is 512.
            out_size (int, optional): Dimension of the output layer. Default is 1.
            kernel_size (int, optional): Kernel size for the convolutional layers. Default is 5.
            n_groups (int, optional): Number of groups for group normalization. Default is 8.
            downsample (bool, optional): Whether to apply downsampling. If we do not downsample, we predict one value
                per timestep. Default is True.
            final_activation (nn.Module, optional): Activation function to use after the final block.
                Default is nn.Identity().
            predict_scale_of_condition (bool, optional): Whether to predict the scale of the condition. Default is True.

        """
        super().__init__()
        all_dims = [input_size] + list(down_sizes)

        # Embed sigma.
        if sigma_embedding is not None:
            sigma_embedding_size = sigma_embedding.embedding_size
        else:
            sigma_embedding_size = 0
        condition_size = sigma_embedding_size
        self.sigma_embedding = sigma_embedding

        # Embed global conditions from extra_inputs.
        if global_condition_embedding is not None:
            self.global_condition_embedding = global_condition_embedding
            condition_size += self.global_condition_embedding.embedding_size

        # Define the input and output sizes for the layers
        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        # Define the local condition encoder. Local conditions are used to condition the model on local features.
        # The local conditions are added to the first down encoder.
        local_condition_encoder = None
        if local_condition_size is not None:
            _, output_channels = in_out[0]
            input_channels = local_condition_size
            local_condition_encoder = torch.nn.ModuleList(
                [
                    # local condition encoder for down encoder
                    ConditionalResidualBlock1D(
                        input_channels,
                        output_channels,
                        condition_size=condition_size,
                        kernel_size=kernel_size,
                        n_groups=n_groups,
                        predict_scale_of_condition=predict_scale_of_condition,
                    )
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
                        Downsample1d(output_channels)
                        if not is_last and downsample
                        else torch.nn.Identity(),  # In Janner et al, the last downsample is applied
                    ]
                )
            )

        # Define the mid modules of the U-Net as two residual blocks that each halve the number of channels.
        latent_size = all_dims[-1]
        self.mid_modules = torch.nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            latent_size,
                            latent_size // 2,
                            condition_size=condition_size,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            predict_scale_of_condition=predict_scale_of_condition,
                        ),
                        Downsample1d(latent_size // 2) if downsample else torch.nn.Identity(),
                    ]
                ),
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            latent_size // 2,
                            latent_size // 4,
                            condition_size=condition_size,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            predict_scale_of_condition=predict_scale_of_condition,
                        ),
                        Downsample1d(latent_size // 4) if downsample else torch.nn.Identity(),
                    ]
                ),
            ]
        )

        # Define the a block that adjusts the shape, depending on wether we downsample or not.
        if downsample:
            self.conv_and_sum = Conv_and_Sum(latent_size // 4)  # sums over last (time) dimension

        # Define the final block of the U-Net as a convolutional block followed by a fully connected layer.
        final_block = torch.nn.Sequential(
            nn.Linear(latent_size // 4 + condition_size, fully_connected_size),
            nn.Mish(),
            nn.Linear(fully_connected_size, out_size),
        )
        # if downsample:
        # final_block = torch.nn.Sequential(
        #     nn.Flatten(start_dim=1),
        #     nn.Linear(latent_size//4, fully_connected_size),
        #     nn.Mish(),
        #     nn.Linear(fully_connected_size, 1),
        # )
        # If we do not downsample, we predict one value per timestep.
        # else:
        #     final_block = torch.nn.Sequential(
        #         Conv1dBlock(latent_size//4, fully_connected_size, kernel_size=kernel_size),
        #         Conv1dBlock(fully_connected_size, 1, kernel_size=kernel_size, n_groups=0),
        #         nn.Flatten(start_dim=1),
        #     )
        self.downsample = downsample

        self.local_condition_encoder = local_condition_encoder
        self.down_modules = down_modules
        self.final_block = final_block
        self.final_activation = final_activation

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
            extra_inputs (tensordict.TensorDict):
                - TensorDict with key "local_condition" and tensor value (batch_size, horizon, local_condition_size).
                - TensorDict with key "global_condition" and tensor value (batch_size,global_condition_size).

        Returns:
            torch.Tensor: Model output. (batch_size, horizon) if predict_one_value_per_timestep else (batch, 1).

        """
        # swap from (batch, horizon, features) to (batch, features, horizon) to convolve over time, not features.
        noisy_sample = noisy_sample.swapaxes(-1, -2)

        # embed sigma
        if sigma is not None and self.sigma_embedding is not None:
            if sigma.shape == torch.Size([]):
                sigma = sigma.expand(noisy_sample.shape[0], 1, 1)
            global_feature = self.sigma_embedding(sigma.squeeze(-1))  # (batch, sigma_embedding_size)
        else:
            global_feature = None

        # global features
        if (
            extra_inputs is not None
            and "global_condition" in extra_inputs
            and extra_inputs["global_condition"] is not None
        ):
            global_condition = self.global_condition(extra_inputs["global_condition"])
            if global_feature is not None:
                global_feature = torch.cat([global_feature, global_condition], dim=-1)
            else:
                global_feature = global_condition

        # encode local features
        h_local = None
        if (
            extra_inputs is not None
            and "local_condition" in extra_inputs
            and extra_inputs["local_condition"] is not None
            and self.local_condition_encoder is not None
        ):
            local_condition = extra_inputs["local_condition"]
            local_condition = local_condition.swapaxes(-1, -2)
            h_local = self.local_condition_encoder[0](local_condition, global_feature)

        # down modules
        x = noisy_sample
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):  # type: ignore
            x = resnet(x, global_feature)
            if idx == 0 and h_local is not None:
                x = x + h_local
            x = resnet2(x, global_feature)
            x = downsample(x)

        # mid modules
        for resnet, downsample in self.mid_modules:  # type: ignore
            x = resnet(x, global_feature)
            x = downsample(x)

        # adjust shape
        if self.downsample:
            x = self.conv_and_sum(x)  # (batch, latent_size//4)
            if global_feature is not None:
                x = torch.cat([x, global_feature], dim=-1)  # (batch, latent_size//4 + global_condition_size)
        else:
            x = x.swapaxes(-1, -2)  # (batch, horizon, latent_size//4)
            if global_feature is not None:
                x = torch.cat(
                    [x, global_feature.unsqueeze(1).expand(-1, x.shape[1], -1)], dim=-1
                )  # (batch, horizon, latent_size//4 + global_condition_size)

        # final block
        x = self.final_block(x)  # (batch, out_dim) or (batch, horizon, out_dim)

        # if not self.downsample:
        #     x = x.squeeze(-1) # (batch, horizon, out_dim)

        # apply activation
        if self.final_activation is not None:
            x = self.final_activation(x)

        return x


class NonTemporalHalfUNetFilm(TemporalHalfUNetFilm):
    """NonTemporalHalfUNet is a subclass of TemporalHalfUNet.

    This is a version of the HalfUNet that does not consider the temporal dimension. There is no flow of information
    between feature vectors of individual timesteps. The model therefore applies the same neural network to every
    timestep. It predicts one value for every timestep.
    """

    def __init__(
        self,
        input_size: int,
        sigma_embedding: Optional[BaseEmbedding] = None,
        global_condition_embedding: Optional[BaseEmbedding] = None,
        local_condition_size: Optional[int] = None,
        down_sizes: List[int] = [512, 1024, 2048],
        fully_connected_size: int = 512,
        out_size: int = 1,
        n_groups: int = 8,
        final_activation: nn.Module = nn.Identity(),
        predict_scale_of_condition: bool = True,
    ):
        """Initialize the HalfUNet class.

        Args:
            input_size (int): The size of the input.
            sigma_embedding (nn.Module): The sigma embedding.
            global_condition_embedding (nn.Module): The global condition embedding.
            local_condition_size (int, optional): The size of the local condition. Default is None.
            global_condition_size (int, optional): The size of the global condition. Default is None.
            down_sizes (List[int], optional): List of sizes for the downsampling layers. Default is [512, 1024, 2048].
            fully_connected_size (int, optional): Dimension of the fully connected layer. Default is 512.
            out_size (int, optional): Dimension of the output layer. Default is 1.
            n_groups (int, optional): Number of groups for group normalization. Default is 8.
            final_activation (nn.Module, optional): Activation function to use. Default is nn.Identity().
            predict_scale_of_condition (bool, optional): Whether to predict the scale of the condition. Default is True.

        """
        super().__init__(
            input_size=input_size,
            sigma_embedding=sigma_embedding,
            global_condition_embedding=global_condition_embedding,
            local_condition_size=local_condition_size,
            down_sizes=down_sizes,
            fully_connected_size=fully_connected_size,
            out_size=out_size,
            kernel_size=1,  # kernel size 1 to prevent information flow between timesteps
            n_groups=1,  # LayerNorm
            downsample=False,  # no downsampling to prevent information flow between timesteps
            final_activation=final_activation,
            predict_scale_of_condition=predict_scale_of_condition,
        )

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
            extra_inputs (tensordict.TensorDict):
                - TensorDict with key "local_condition" and tensor value (batch_size, horizon, local_condition_size).
                - TensorDict with key "global_condition" and tensor value (batch_size,global_condition_size).

        Returns:
            torch.Tensor: Model output. (batch_size, horizon).

        """
        # The network outputs one value per timestep. We sum over the last dimension to get one value per sample.
        return super().forward(noisy_sample, sigma, extra_inputs)  # .sum(-1, keepdim=True)
