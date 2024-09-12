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

from PIL import Image, ImageEnhance, ImageOps
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union
import numbers
import numpy as np

try:
    import accimage
except ImportError:
    accimage = None

def _is_pil_image(img: Any) -> bool:
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)

def pad(
    img: Image.Image,
    padding: Union[int, List[int], Tuple[int, ...]],
    fill: Optional[Union[float, List[float], Tuple[float, ...]]] = 0,
    padding_mode: Literal["constant", "edge", "reflect", "symmetric"] = "constant",
) -> Image.Image:

    if not _is_pil_image(img):
        raise TypeError(f"img should be PIL Image. Got {type(img)}")

    if not isinstance(padding, (numbers.Number, tuple, list)):
        raise TypeError("Got inappropriate padding arg")
    if fill is not None and not isinstance(fill, (numbers.Number, tuple, list)):
        raise TypeError("Got inappropriate fill arg")
    if not isinstance(padding_mode, str):
        raise TypeError("Got inappropriate padding_mode arg")

    if isinstance(padding, list):
        padding = tuple(padding)

    if isinstance(padding, tuple) and len(padding) not in [1, 2, 4]:
        raise ValueError(f"Padding must be an int or a 1, 2, or 4 element tuple, not a {len(padding)} element tuple")

    if isinstance(padding, tuple) and len(padding) == 1:
        # Compatibility with `functional_tensor.pad`
        padding = padding[0]

    if padding_mode not in ["constant", "edge", "reflect", "symmetric"]:
        raise ValueError("Padding mode should be either constant, edge, reflect or symmetric")

    if padding_mode == "constant":
        opts = _parse_fill(fill, img, name="fill")
        if img.mode == "P":
            palette = img.getpalette()
            image = ImageOps.expand(img, border=padding, **opts)
            image.putpalette(palette)
            return image

        return ImageOps.expand(img, border=padding, **opts)
    else:
        if isinstance(padding, int):
            pad_left = pad_right = pad_top = pad_bottom = padding
        if isinstance(padding, tuple) and len(padding) == 2:
            pad_left = pad_right = padding[0]
            pad_top = pad_bottom = padding[1]
        if isinstance(padding, tuple) and len(padding) == 4:
            pad_left = padding[0]
            pad_top = padding[1]
            pad_right = padding[2]
            pad_bottom = padding[3]

        p = [pad_left, pad_top, pad_right, pad_bottom]
        cropping = -np.minimum(p, 0)

        if cropping.any():
            crop_left, crop_top, crop_right, crop_bottom = cropping
            img = img.crop((crop_left, crop_top, img.width - crop_right, img.height - crop_bottom))

        pad_left, pad_top, pad_right, pad_bottom = np.maximum(p, 0)

        if img.mode == "P":
            palette = img.getpalette()
            img = np.asarray(img)
            img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), mode=padding_mode)
            img = Image.fromarray(img)
            img.putpalette(palette)
            return img

        img = np.asarray(img)
        # RGB image
        if len(img.shape) == 3:
            img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), padding_mode)
        # Grayscale image
        if len(img.shape) == 2:
            img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), padding_mode)

        return Image.fromarray(img)

def get_image_num_channels(img: Any) -> int:
    if _is_pil_image(img):
        if hasattr(img, "getbands"):
            return len(img.getbands())
        else:
            return img.channels
    raise TypeError(f"Unexpected type {type(img)}")


def _parse_fill(
    fill: Optional[Union[float, List[float], Tuple[float, ...]]],
    img: Image.Image,
    name: str = "fillcolor",
) -> Dict[str, Optional[Union[float, List[float], Tuple[float, ...]]]]:

    # Process fill color for affine transforms
    num_channels = get_image_num_channels(img)
    if fill is None:
        fill = 0
    if isinstance(fill, (int, float)) and num_channels > 1:
        fill = tuple([fill] * num_channels)
    if isinstance(fill, (list, tuple)):
        if len(fill) != num_channels:
            msg = "The number of elements in 'fill' does not match the number of channels of the image ({} != {})"
            raise ValueError(msg.format(len(fill), num_channels))

        fill = tuple(fill)

    if img.mode != "F":
        if isinstance(fill, (list, tuple)):
            fill = tuple(int(x) for x in fill)
        else:
            fill = int(fill)

    return {name: fill}
