# Copyright (C) 2024 Jiang Xin
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import warnings
from typing import Callable, List, Optional, Sequence, Tuple, Union

import jittor as jt
from ..utils import _make_ntuple


class ConvNormActivation(jt.nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]] = 3,
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Optional[Union[int, Tuple[int, ...], str]] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., jt.nn.Module]] = jt.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., jt.nn.Module]] = jt.nn.ReLU,
        dilation: Union[int, Tuple[int, ...]] = 1,
        bias: Optional[bool] = None,
        conv_layer: Callable[..., jt.nn.Module] = jt.nn.Conv2d,
    ) -> None:

        if padding is None:
            if isinstance(kernel_size, int) and isinstance(dilation, int):
                padding = (kernel_size - 1) // 2 * dilation
            else:
                _conv_dim = len(kernel_size) if isinstance(kernel_size, Sequence) else len(dilation)
                kernel_size = _make_ntuple(kernel_size, _conv_dim)
                dilation = _make_ntuple(dilation, _conv_dim)
                padding = tuple((kernel_size[i] - 1) // 2 * dilation[i] for i in range(_conv_dim))
        if bias is None:
            bias = norm_layer is None

        layers = [
            conv_layer(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        ]

        if norm_layer is not None:
            layers.append(norm_layer(out_channels))

        if activation_layer is not None:
            params = {}
            layers.append(activation_layer(**params))
        super().__init__(*layers)
        self.out_channels = out_channels
        if self.__class__ == ConvNormActivation:
            warnings.warn(
                "Don't use ConvNormActivation directly, please use Conv2dNormActivation and Conv3dNormActivation instead."
            )


class Conv2dNormActivation(ConvNormActivation):
    """
    Configurable block used for Convolution2d-Normalization-Activation blocks.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the Convolution-Normalization-Activation block
        kernel_size: (int, optional): Size of the convolving kernel. Default: 3
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all four sides of the input. Default: None, in which case it will be calculated as ``padding = (kernel_size - 1) // 2 * dilation``
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        norm_layer (Callable[..., jt.nn.Module], optional): Norm layer that will be stacked on top of the convolution layer. If ``None`` this layer won't be used. Default: ``jt.nn.BatchNorm2d``
        activation_layer (Callable[..., jt.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the conv layer. If ``None`` this layer won't be used. Default: ``jt.nn.ReLU``
        dilation (int): Spacing between kernel elements. Default: 1
        inplace (bool): Parameter for the activation layer, which can optionally do the operation in-place. Default ``True``
        bias (bool, optional): Whether to use bias in the convolution layer. By default, biases are included if ``norm_layer is None``.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Optional[Union[int, Tuple[int, int], str]] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., jt.nn.Module]] = jt.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., jt.nn.Module]] = jt.nn.ReLU,
        dilation: Union[int, Tuple[int, int]] = 1,
        bias: Optional[bool] = None,
    ) -> None:

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups,
            norm_layer,
            activation_layer,
            dilation,
            bias,
            jt.nn.Conv2d,
        )


class Conv3dNormActivation(ConvNormActivation):
    """
    Configurable block used for Convolution3d-Normalization-Activation blocks.

    Args:
        in_channels (int): Number of channels in the input video.
        out_channels (int): Number of channels produced by the Convolution-Normalization-Activation block
        kernel_size: (int, optional): Size of the convolving kernel. Default: 3
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all four sides of the input. Default: None, in which case it will be calculated as ``padding = (kernel_size - 1) // 2 * dilation``
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        norm_layer (Callable[..., jt.nn.Module], optional): Norm layer that will be stacked on top of the convolution layer. If ``None`` this layer won't be used. Default: ``jt.nn.BatchNorm3d``
        activation_layer (Callable[..., jt.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the conv layer. If ``None`` this layer won't be used. Default: ``jt.nn.ReLU``
        dilation (int): Spacing between kernel elements. Default: 1
        inplace (bool): Parameter for the activation layer, which can optionally do the operation in-place. Default ``True``
        bias (bool, optional): Whether to use bias in the convolution layer. By default, biases are included if ``norm_layer is None``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Optional[Union[int, Tuple[int, int, int], str]] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., jt.nn.Module]] = jt.nn.BatchNorm3d,
        activation_layer: Optional[Callable[..., jt.nn.Module]] = jt.nn.ReLU,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        bias: Optional[bool] = None,
    ) -> None:

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups,
            norm_layer,
            activation_layer,
            dilation,
            bias,
            jt.nn.Conv3d,
        )


# class SqueezeExcitation(jt.nn.Module):
#     """
#     This block implements the Squeeze-and-Excitation block from https://arxiv.org/abs/1709.01507 (see Fig. 1).
#     Parameters ``activation``, and ``scale_activation`` correspond to ``delta`` and ``sigma`` in eq. 3.

#     Args:
#         input_channels (int): Number of channels in the input image
#         squeeze_channels (int): Number of squeeze channels
#         activation (Callable[..., jt.nn.Module], optional): ``delta`` activation. Default: ``jt.nn.ReLU``
#         scale_activation (Callable[..., jt.nn.Module]): ``sigma`` activation. Default: ``jt.nn.Sigmoid``
#     """

#     def __init__(
#         self,
#         input_channels: int,
#         squeeze_channels: int,
#         activation: Callable[..., jt.nn.Module] = jt.nn.ReLU,
#         scale_activation: Callable[..., jt.nn.Module] = jt.nn.Sigmoid,
#     ) -> None:
#         super().__init__()
#         self.avgpool = jt.nn.AdaptiveAvgPool2d(1)
#         self.fc1 = jt.nn.Conv2d(input_channels, squeeze_channels, 1)
#         self.fc2 = jt.nn.Conv2d(squeeze_channels, input_channels, 1)
#         self.activation = activation()
#         self.scale_activation = scale_activation()

#     def _scale(self, input):
#         scale = self.avgpool(input)
#         scale = self.fc1(scale)
#         scale = self.activation(scale)
#         scale = self.fc2(scale)
#         return self.scale_activation(scale)

#     def forward(self, input):
#         scale = self._scale(input)
#         return scale * input


# class MLP(jt.nn.Sequential):
#     """This block implements the multi-layer perceptron (MLP) module.

#     Args:
#         in_channels (int): Number of channels of the input
#         hidden_channels (List[int]): List of the hidden channel dimensions
#         norm_layer (Callable[..., jt.nn.Module], optional): Norm layer that will be stacked on top of the linear layer. If ``None`` this layer won't be used. Default: ``None``
#         activation_layer (Callable[..., jt.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the linear layer. If ``None`` this layer won't be used. Default: ``jt.nn.ReLU``
#         inplace (bool, optional): Parameter for the activation layer, which can optionally do the operation in-place.
#             Default is ``None``, which uses the respective default values of the ``activation_layer`` and Dropout layer.
#         bias (bool): Whether to use bias in the linear layer. Default ``True``
#         dropout (float): The probability for the dropout layer. Default: 0.0
#     """

#     def __init__(
#         self,
#         in_channels: int,
#         hidden_channels: List[int],
#         norm_layer: Optional[Callable[..., jt.nn.Module]] = None,
#         activation_layer: Optional[Callable[..., jt.nn.Module]] = jt.nn.ReLU,
#         inplace: Optional[bool] = None,
#         bias: bool = True,
#         dropout: float = 0.0,
#     ):
#         # The addition of `norm_layer` is inspired from the implementation of jtMultimodal:
#         # https://github.com/facebookresearch/multimodal/blob/5dec8a/jtmultimodal/modules/layers/mlp.py
#         params = {} if inplace is None else {"inplace": inplace}

#         layers = []
#         in_dim = in_channels
#         for hidden_dim in hidden_channels[:-1]:
#             layers.append(jt.nn.Linear(in_dim, hidden_dim, bias=bias))
#             if norm_layer is not None:
#                 layers.append(norm_layer(hidden_dim))
#             layers.append(activation_layer(**params))
#             layers.append(jt.nn.Dropout(dropout, **params))
#             in_dim = hidden_dim

#         layers.append(jt.nn.Linear(in_dim, hidden_channels[-1], bias=bias))
#         layers.append(jt.nn.Dropout(dropout, **params))

#         super().__init__(*layers)
#         _log_api_usage_once(self)


# class Permute(jt.nn.Module):
#     """This module returns a view of the tensor input with its dimensions permuted.

#     Args:
#         dims (List[int]): The desired ordering of dimensions
#     """

#     def __init__(self, dims: List[int]):
#         super().__init__()
#         self.dims = dims

#     def forward(self, x: Tensor) -> Tensor:
#         return jt.permute(x, self.dims)

class SqueezeExcitation(jt.nn.Module):
    """
    This block implements the Squeeze-and-Excitation block from https://arxiv.org/abs/1709.01507 (see Fig. 1).
    Parameters ``activation``, and ``scale_activation`` correspond to ``delta`` and ``sigma`` in eq. 3.

    Args:
        input_channels (int): Number of channels in the input image
        squeeze_channels (int): Number of squeeze channels
        activation (Callable[..., jt.nn.Module], optional): ``delta`` activation. Default: ``jt.nn.ReLU``
        scale_activation (Callable[..., jt.nn.Module]): ``sigma`` activation. Default: ``jt.nn.Sigmoid``
    """

    def __init__(
        self,
        input_channels: int,
        squeeze_channels: int,
        activation: Callable[..., jt.nn.Module] = jt.nn.ReLU,
        scale_activation: Callable[..., jt.nn.Module] = jt.nn.Sigmoid,
    ) -> None:
        super().__init__()
        self.avgpool = jt.nn.AdaptiveAvgPool2d(1)
        self.fc1 = jt.nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = jt.nn.Conv2d(squeeze_channels, input_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, input):
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def execute(self, input):
        scale = self._scale(input)
        return scale * input