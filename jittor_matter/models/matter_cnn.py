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

from jittor import nn

from .mobilenetv3 import MobileNetV3LargeEncoder
from .lraspp import LRASPP
from .decoder_cnn import CNNDecoder, Projection


class MattingNetwork(nn.Module):
    def __init__(self,
                 variant: str = 'mobilenetv3',
                 refiner: str = 'deep_guided_filter',
                 pretrained_backbone: bool = True,
                 dropout=0.1,
                 ):
        super().__init__()
        assert variant in ['mobilenetv3',
                           'mobilenetv3small', 'resnet18', 'resnet50']
        assert refiner in ['fast_guided_filter', 'deep_guided_filter']

        if variant == 'mobilenetv3':
            self.backbone = MobileNetV3LargeEncoder(pretrained_backbone)
            self.aspp = LRASPP(960, 128)
            self.decoder = CNNDecoder(
                enc_channels=[16, 24, 40, 128], hr_channels=32)

        self.project_mat = Projection(32, 5)

    def execute(self,
                x,
                src,
                downsample_ratio: float = 1,
                segmentation_pass: bool = False):

        if downsample_ratio != 1:
            x_sm = self._interpolate(x, scale_factor=downsample_ratio)
            src_sm = self._interpolate(src, scale_factor=downsample_ratio)
        else:
            x_sm = x
            src_sm = src

        f1, f2, f3, f4 = self.backbone(x_sm)
        # print('######## f4.shape',f4.shape)

        f4 = self.aspp(f4)

        hid = self.decoder(src_sm, f1, f2, f3, f4)

        fgr_residual, pha, details = self.project_mat(
            hid).split([3, 1, 1], dim=1)
        # fgr = fgr_residual + src
        fgr = fgr_residual
        fgr = fgr.clamp(0., 1.)
        pha = pha.clamp(0., 1.)
        details = details.clamp(0., 1.)
        return [fgr, pha, details]

    def _interpolate(self, x, scale_factor):
        if x.ndim == 5:
            B, T = x.shape[:2]
            x = nn.interpolate(x.flatten(0, 1), scale_factor=scale_factor,
                               mode='bilinear', align_corners=False)
            x = x.unflatten(0, (B, T))
        else:
            x = nn.interpolate(x, scale_factor=scale_factor,
                               mode='bilinear', align_corners=False)
        return x
