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

import functools
import inspect
import warnings
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, Union
from jittor import nn
import jittor as jt

from .._utils import sequence_to_str


def padding(im, patch_size, fill_value=0):
    # make the image sizes divisible by patch_size
    H, W = im.size(2), im.size(3)
    pad_h, pad_w = 0, 0
    if H % patch_size > 0:
        pad_h = patch_size - (H % patch_size)
    if W % patch_size > 0:
        pad_w = patch_size - (W % patch_size)
    im_padded = im
    if pad_h > 0 or pad_w > 0:
        im_padded = nn.pad(im, (0, pad_w, 0, pad_h), value=fill_value)
    return im_padded


def unpadding(y, target_size):
    H, W = target_size
    H_pad, W_pad = y.size(2), y.size(3)
    # crop predictions on extra pixels coming from padding
    extra_h = H_pad - H
    extra_w = W_pad - W
    if extra_h > 0:
        y = y[:, :, :-extra_h]
    if extra_w > 0:
        y = y[:, :, :, :-extra_w]
    return y


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


# class IntermediateLayerGetter(nn.ModuleDict):
#     """
#     Module wrapper that returns intermediate layers from a model

#     It has a strong assumption that the modules have been registered
#     into the model in the same order as they are used.
#     This means that one should **not** reuse the same nn.Module
#     twice in the forward if you want this to work.

#     Additionally, it is only able to query submodules that are directly
#     assigned to the model. So if `model` is passed, `model.feature1` can
#     be returned, but not `model.feature1.layer2`.

#     Args:
#         model (nn.Module): model on which we will extract the features
#         return_layers (Dict[name, new_name]): a dict containing the names
#             of the modules for which the activations will be returned as
#             the key of the dict, and the value of the dict is the name
#             of the returned activation (which the user can specify).

#     Examples::

#         >>> m = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
#         >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
#         >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
#         >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
#         >>> out = new_m(torch.rand(1, 3, 224, 224))
#         >>> print([(k, v.shape) for k, v in out.items()])
#         >>>     [('feat1', torch.Size([1, 64, 56, 56])),
#         >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
#     """

#     _version = 2
#     __annotations__ = {
#         "return_layers": Dict[str, str],
#     }

#     def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
#         if not set(return_layers).issubset([name for name, _ in model.named_children()]):
#             raise ValueError("return_layers are not present in model")
#         orig_return_layers = return_layers
#         return_layers = {str(k): str(v) for k, v in return_layers.items()}
#         layers = OrderedDict()
#         for name, module in model.named_children():
#             layers[name] = module
#             if name in return_layers:
#                 del return_layers[name]
#             if not return_layers:
#                 break

#         super().__init__(layers)
#         self.return_layers = orig_return_layers

#     def forward(self, x):
#         out = OrderedDict()
#         for name, module in self.items():
#             x = module(x)
#             if name in self.return_layers:
#                 out_name = self.return_layers[name]
#                 out[out_name] = x
#         return out


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


D = TypeVar("D")


def kwonly_to_pos_or_kw(fn: Callable[..., D]) -> Callable[..., D]:
    """Decorates a function that uses keyword only parameters to also allow them being passed as positionals.

    For example, consider the use case of changing the signature of ``old_fn`` into the one from ``new_fn``:

    .. code::

        def old_fn(foo, bar, baz=None):
            ...

        def new_fn(foo, *, bar, baz=None):
            ...

    Calling ``old_fn("foo", "bar, "baz")`` was valid, but the same call is no longer valid with ``new_fn``. To keep BC
    and at the same time warn the user of the deprecation, this decorator can be used:

    .. code::

        @kwonly_to_pos_or_kw
        def new_fn(foo, *, bar, baz=None):
            ...

        new_fn("foo", "bar, "baz")
    """
    params = inspect.signature(fn).parameters

    try:
        keyword_only_start_idx = next(
            idx for idx, param in enumerate(params.values()) if param.kind == param.KEYWORD_ONLY
        )
    except StopIteration:
        raise TypeError(
            f"Found no keyword-only parameter on function '{fn.__name__}'") from None

    keyword_only_params = tuple(inspect.signature(fn).parameters)[
        keyword_only_start_idx:]

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> D:
        args, keyword_only_args = args[:keyword_only_start_idx], args[keyword_only_start_idx:]
        if keyword_only_args:
            keyword_only_kwargs = dict(
                zip(keyword_only_params, keyword_only_args))
            warnings.warn(
                f"Using {sequence_to_str(tuple(keyword_only_kwargs.keys()), separate_last='and ')} as positional "
                f"parameter(s) is deprecated since 0.13 and may be removed in the future. Please use keyword parameter(s) "
                f"instead."
            )
            kwargs.update(keyword_only_kwargs)

        return fn(*args, **kwargs)

    return wrapper


M = TypeVar("M", bound=nn.Module)
V = TypeVar("V")


def _ovewrite_named_param(kwargs: Dict[str, Any], param: str, new_value: V) -> None:
    if param in kwargs:
        if kwargs[param] != new_value:
            raise ValueError(
                f"The parameter '{param}' expected value {new_value} but got {kwargs[param]} instead.")
    else:
        kwargs[param] = new_value


def _ovewrite_value_param(param: str, actual: Optional[V], expected: V) -> V:
    if actual is not None:
        if actual != expected:
            raise ValueError(
                f"The parameter '{param}' expected value {expected} but got {actual} instead.")
    return expected


class _ModelURLs(dict):
    def __getitem__(self, item):
        warnings.warn(
            "Accessing the model URLs via the internal dictionary of the module is deprecated since 0.13 and may "
            "be removed in the future. Please access them via the appropriate Weights Enum instead."
        )
        return super().__getitem__(item)


def erode(x, k_size=7, stride=1, padding=3):
    _kernel = jt.zeros(1, k_size, k_size)
    _kernel[:, k_size//2, k_size//2] = 1
    return jt.nn.conv2d(x, _kernel, bias=None, stride=stride, padding=padding, dilation=1, groups=1)
