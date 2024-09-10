import jittor as jt
import jittor.nn as nn


class IBNorm(nn.Module):
    """ Combine Instance Norm and Batch Norm into One Layer
    """

    def __init__(self, in_channels):
        super(IBNorm, self).__init__()
        in_channels = in_channels
        self.bnorm_channels = int(in_channels / 2)
        self.inorm_channels = in_channels - self.bnorm_channels
        self.bnorm = nn.BatchNorm2d(self.bnorm_channels, affine=True)
        self.inorm = nn.InstanceNorm2d(self.inorm_channels, affine=False)

    def execute(self, x):
        bn_x = self.bnorm(x[:, :self.bnorm_channels, ...].contiguous())
        in_x = self.inorm(x[:, self.bnorm_channels:, ...].contiguous())
        return jt.cat((bn_x, in_x), 1)


class Conv2dIBNormRelu(nn.Module):
    """ Convolution + IBNorm + ReLu
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 with_ibn=True, with_relu=True):
        super(Conv2dIBNormRelu, self).__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, dilation=dilation,
                      groups=groups, bias=bias)
        ]

        if with_ibn:
            layers.append(IBNorm(out_channels))
        if with_relu:
            layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)

    def execute(self, x):
        return self.layers(x)


class CNNDecoder(nn.Module):
    """ High Resolution Branch of MODNet
    """

    def __init__(self, enc_channels, hr_channels):
        super(CNNDecoder, self).__init__()

        self.tohr_enc2x = Conv2dIBNormRelu(
            enc_channels[0], hr_channels, 1, stride=1, padding=0)
        self.tohr_enc4x = Conv2dIBNormRelu(
            enc_channels[1], hr_channels, 1, stride=1, padding=0)
        self.tohr_enc8x = Conv2dIBNormRelu(
            enc_channels[2], hr_channels, 1, stride=1, padding=0)
        self.tohr_enc16x = Conv2dIBNormRelu(
            enc_channels[3], hr_channels, 1, stride=1, padding=0)

        self.conv_hr8x = nn.Sequential(
            Conv2dIBNormRelu(2 * hr_channels, 2 * hr_channels,
                             3, stride=1, padding=1),
            Conv2dIBNormRelu(2 * hr_channels, 2 * hr_channels,
                             3, stride=1, padding=1),
            Conv2dIBNormRelu(2 * hr_channels, hr_channels,
                             3, stride=1, padding=1),
        )

        self.conv_hr4x = nn.Sequential(
            Conv2dIBNormRelu(2 * hr_channels, 2 * hr_channels,
                             3, stride=1, padding=1),
            Conv2dIBNormRelu(2 * hr_channels, 2 * hr_channels,
                             3, stride=1, padding=1),
            Conv2dIBNormRelu(2 * hr_channels, hr_channels,
                             3, stride=1, padding=1),
        )

        self.conv_hr2x = nn.Sequential(
            Conv2dIBNormRelu(2 * hr_channels, 2 * hr_channels,
                             5, stride=1, padding=2),
            Conv2dIBNormRelu(2 * hr_channels, hr_channels,
                             3, stride=1, padding=1),
            Conv2dIBNormRelu(hr_channels, hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(hr_channels, hr_channels, 3, stride=1, padding=1),
        )

        self.conv_hr = nn.Sequential(
            Conv2dIBNormRelu(hr_channels + 3, hr_channels,
                             5, stride=1, padding=2),
            Conv2dIBNormRelu(hr_channels, hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(hr_channels, hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(hr_channels, 32, kernel_size=3, stride=1,
                             padding=1, with_ibn=False, with_relu=False),
        )

    def execute(self, img, f1, f2, f3, f4):
        '''
            s0 : src img e.g. size (960 720)
            #### F1:torch.Size([1, 16, 480, 360])  2x
            #### F2:torch.Size([1, 24, 240, 180])  4x
            #### F3:torch.Size([1, 40, 120, 90])   8x
            #### F4:torch.Size([1, 128, 60, 45])   16x
        '''
        H, W = img.shape[-2:]
        enc2x = self.tohr_enc2x(f1)
        enc4x = self.tohr_enc4x(f2)
        enc8x = self.tohr_enc8x(f3)
        enc16x = self.tohr_enc16x(f4)

        size_8x = enc8x.shape[-2:]
        size_4x = enc4x.shape[-2:]
        size_2x = enc2x.shape[-2:]

        hr = nn.interpolate(enc16x, size_8x, mode='bilinear',
                            align_corners=False)  # 8x

        hr = self.conv_hr8x(jt.cat((hr, enc8x), dim=1))

        hr = nn.interpolate(
            hr, size=size_4x, mode='bilinear', align_corners=False)
        hr = self.conv_hr4x(jt.cat((hr, enc4x), dim=1))

        hr = nn.interpolate(
            hr, size=size_2x, mode='bilinear', align_corners=False)
        hr = self.conv_hr2x(jt.cat((hr, enc2x), dim=1))

        hr = nn.interpolate(
            hr, size=(H, W), mode='bilinear', align_corners=False)
        hr = self.conv_hr(jt.cat((hr, img), dim=1))
        # pred = torch.sigmoid(hr)
        return hr


class Projection(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward_single_frame(self, x):
        return jt.sigmoid(self.conv(x))

    def forward_time_series(self, x):
        B, T = x.shape[:2]
        return self.conv(x.flatten(0, 1)).unflatten(0, (B, T))

    def execute(self, x):
        if x.ndim == 5:
            return self.forward_time_series(x)
        else:
            return self.forward_single_frame(x)
