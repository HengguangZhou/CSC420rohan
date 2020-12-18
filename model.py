import math
from torch import nn
import torch

# Reference: CSC420 Lecture 8 Deep Learing Tutorial, Hands-on Part

# Corresponding degradation model FSRCNN
class FLRCNN(nn.Module):
    def __init__(self, scale, d=56, s=12, m=4, input_channels=1):
        super(FLRCNN, self).__init__()
        self.extract = nn.Sequential(*[nn.Conv2d(input_channels, d, kernel_size=5, padding=2), nn.PReLU(d)])
        self.shrink = nn.Sequential(*[nn.Conv2d(d, s, kernel_size=1), nn.PReLU(s)])
        self.mapping = nn.Sequential(*([nn.Conv2d(s, s, kernel_size=3, padding=1) for _ in range(m)] + [nn.PReLU(s)]))
        self.expand = nn.Sequential(*[nn.Conv2d(s, d, kernel_size=1), nn.PReLU(d)])
        self.downconv = nn.Conv2d(d, input_channels, kernel_size=9, stride=scale, padding=4)

    def forward(self, x):
        x = self.extract(x)
        x = self.shrink(x)
        x = self.mapping(x)
        x = self.expand(x)
        return self.downconv(x)

# Model for FSRCNN
class FSRCNN(nn.Module):
    def __init__(self, scale, d=56, s=12, m=4, input_channels=1):
        super(FSRCNN, self).__init__()
        self.extract = nn.Sequential(*[nn.Conv2d(input_channels, d, kernel_size=5, padding=2), nn.PReLU(d)])
        self.shrink = nn.Sequential(*[nn.Conv2d(d, s, kernel_size=1), nn.PReLU(s)])
        self.mapping = nn.Sequential(*([nn.Conv2d(s, s, kernel_size=3, padding=1) for _ in range(m)] + [nn.PReLU(s)]))
        self.expand = nn.Sequential(*[nn.Conv2d(s, d, kernel_size=1), nn.PReLU(d)])
        self.deconv = nn.ConvTranspose2d(d, input_channels, kernel_size=9, stride=scale, padding=4,
                                         output_padding=scale-1)

    def forward(self, x):
        x = self.extract(x)
        x = self.shrink(x)
        x = self.mapping(x)
        x = self.expand(x)
        return self.deconv(x)

# Model for ESPCN
class ESPCN(nn.Module):
    def __init__(self, scale, input_channels=1):
        super(ESPCN, self).__init__()
        FM1_in, FM1_out, FM1_kernel_size, FM1_stride, FM1_padding = input_channels, 64, 5, 1, 2
        FM2_in, FM2_out, FM2_kernel_size, FM2_stride, FM2_padding = 64, 32, 3, 1, 1
        SP_in, SP_kernel_size, SP_stride, SP_padding = 32, 3, 1, 1
        SP_out = input_channels * pow(scale, 2)
        
        self.featureMap1 = nn.Sequential(
            nn.Conv2d(FM1_in, FM1_out, FM1_kernel_size, FM1_stride, FM1_padding),
            nn.Tanh()
        )

        self.featureMap2 = nn.Sequential(
            nn.Conv2d(FM2_in, FM2_out, FM2_kernel_size, FM2_stride, FM2_padding),
            nn.Tanh()
        )
        
        self.subPixelConv = nn.Sequential(
            nn.Conv2d(SP_in, SP_out, SP_kernel_size, SP_stride, SP_padding),
            nn.PixelShuffle(scale)
        )

    def forward(self, x):
        x = self.featureMap1(x)
        x = self.featureMap2(x)
        x = self.subPixelConv(x)
        return x


# A pre-existing helper that does the reverse operation of torch.nn.PixelShuffle
 
# Source: https://github.com/fangwei123456/PixelUnshuffle-pytorch/blob/master/PixelUnshuffle

def pixel_unshuffle(input, downscale_factor):
    c = input.shape[1]

    kernel = torch.zeros(size=[downscale_factor * downscale_factor * c,
                               1, downscale_factor, downscale_factor],
                         device=input.device)
    for y in range(downscale_factor):
        for x in range(downscale_factor):
            kernel[x + y * downscale_factor::downscale_factor*downscale_factor, 0, y, x] = 1

    return nn.functional.conv2d(input, kernel, stride=downscale_factor, groups=c)

class PixelUnshuffle(nn.Module):

    def __init__(self, downscale_factor):
        super(PixelUnshuffle, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, input):
        return pixel_unshuffle(input, self.downscale_factor)



# Corresponding degradation model for ESPCN
class DESPCN(nn.Module):
    def __init__(self, scale, input_channels=1):
        super(DESPCN, self).__init__()
        
        SPR_in = input_channels * pow(scale, 2)
        SPR_out, SPR_kernel_size, SPR_stride, SPR_padding = 32, 3, 1, 1
        FM2R_in, FM2R_out, FM2R_kernel_size, FM2R_stride, FM2R_padding = 32, 64, 3, 1, 1
        FM1R_in, FM1R_out, FM1R_kernel_size, FM1R_stride, FM1R_padding = 64, input_channels, 5, 1, 2

        self.subPixelConvReverse = nn.Sequential(
            PixelUnshuffle(scale),
            nn.Conv2d(SPR_in, SPR_out, SPR_kernel_size, SPR_stride, SPR_padding)
        )

        self.featureMap2Reverse = nn.Sequential(
            nn.Conv2d(FM2R_in, FM2R_out, FM2R_kernel_size, FM2R_stride, FM2R_padding),
            nn.Tanh()
        )

        self.featureMap1Reverse = nn.Sequential(
            nn.Conv2d(FM1R_in, FM1R_out, FM1R_kernel_size, FM1R_stride, FM1R_padding),
            nn.Tanh()
        )
    def forward(self, x):
            x = self.subPixelConvReverse(x)
            x = self.featureMap2Reverse(x)
            x = self.featureMap1Reverse(x)
            return x


# Model for VDSR (with half numbers of residual layers)
class VDSR(nn.Module):
    def __init__(self, scale, input_channels=1):
        super(VDSR, self).__init__() 
        self.scale = scale
        ReLU_inplace = True
        IL_in, IL_out, IL_kernel_size, IL_stride, IL_padding, IL_bias = 1, 64, 3, 1, 1, False
        RL_in, RL_out, RL_kernel_size, RL_stride, RL_padding, RL_bias = 64, 64, 3, 1, 1, False
        OL_in, OL_out, OL_kernel_size, OL_stride, OL_padding, OL_bias = 64, 1, 3, 1, 1, False

        # Original Paper used 20 layers in total, we are not going to use that many residual layers
        # We used 9 residual layers

        self.inputLayer = nn.Sequential(
            nn.Conv2d(IL_in, IL_out, IL_kernel_size, IL_stride, IL_padding, bias = IL_bias),
            nn.ReLU(ReLU_inplace)
        )
        

        self.resLayer = nn.Sequential(
            nn.Conv2d(RL_in, RL_out, RL_kernel_size, RL_stride, RL_padding, bias = RL_bias),
            nn.ReLU(ReLU_inplace),
            nn.Conv2d(RL_in, RL_out, RL_kernel_size, RL_stride, RL_padding, bias = RL_bias),
            nn.ReLU(ReLU_inplace),
            nn.Conv2d(RL_in, RL_out, RL_kernel_size, RL_stride, RL_padding, bias = RL_bias),
            nn.ReLU(ReLU_inplace),
            nn.Conv2d(RL_in, RL_out, RL_kernel_size, RL_stride, RL_padding, bias = RL_bias),
            nn.ReLU(ReLU_inplace),
            nn.Conv2d(RL_in, RL_out, RL_kernel_size, RL_stride, RL_padding, bias = RL_bias),
            nn.ReLU(ReLU_inplace),
            nn.Conv2d(RL_in, RL_out, RL_kernel_size, RL_stride, RL_padding, bias = RL_bias),
            nn.ReLU(ReLU_inplace),
            nn.Conv2d(RL_in, RL_out, RL_kernel_size, RL_stride, RL_padding, bias = RL_bias),
            nn.ReLU(ReLU_inplace),
            nn.Conv2d(RL_in, RL_out, RL_kernel_size, RL_stride, RL_padding, bias = RL_bias),
            nn.ReLU(ReLU_inplace),
            nn.Conv2d(RL_in, RL_out, RL_kernel_size, RL_stride, RL_padding, bias = RL_bias),
            nn.ReLU(ReLU_inplace)
        )
        self.outputLayer = nn.Sequential(
            nn.Conv2d(OL_in, OL_out, OL_kernel_size, OL_stride, OL_padding, bias = OL_bias),
        )


    def forward(self, x):
        upSample = nn.Upsample(scale_factor=self.scale, mode='bicubic')
        x = upSample(x)
        preserved = x
        output = self.inputLayer(x)
        output = self.resLayer(output)
        output = self.outputLayer(output)
        return preserved + output

# Corresponding degradation model for VDSR (with less residual layers)
class DVDSR(nn.Module):
    def __init__(self, scale, input_channels=1):
        super(DVDSR, self).__init__() 
        self.scale = scale
        ReLU_inplace = True
        IL_in, IL_out, IL_kernel_size, IL_stride, IL_padding, IL_bias = 1, 64, 3, 1, 1, False
        RL_in, RL_out, RL_kernel_size, RL_stride, RL_padding, RL_bias = 64, 64, 3, 1, 1, False
        OL_in, OL_out, OL_kernel_size, OL_stride, OL_padding, OL_bias = 64, 1, 3, 1, 1, False

        # Original Paper used 20 layers in total, we are not going to use that many residual layers

        self.inputLayer = nn.Sequential(
            nn.Conv2d(IL_in, IL_out, IL_kernel_size, IL_stride, IL_padding, bias = IL_bias),
            nn.ReLU(ReLU_inplace)
        )
        

        self.resLayer = nn.Sequential(
            nn.Conv2d(RL_in, RL_out, RL_kernel_size, RL_stride, RL_padding, bias = RL_bias),
            nn.ReLU(ReLU_inplace),
            nn.Conv2d(RL_in, RL_out, RL_kernel_size, RL_stride, RL_padding, bias = RL_bias),
            nn.ReLU(ReLU_inplace),
            nn.Conv2d(RL_in, RL_out, RL_kernel_size, RL_stride, RL_padding, bias = RL_bias),
            nn.ReLU(ReLU_inplace),
            nn.Conv2d(RL_in, RL_out, RL_kernel_size, RL_stride, RL_padding, bias = RL_bias),
            nn.ReLU(ReLU_inplace),
            nn.Conv2d(RL_in, RL_out, RL_kernel_size, RL_stride, RL_padding, bias = RL_bias),
            nn.ReLU(ReLU_inplace),
            nn.Conv2d(RL_in, RL_out, RL_kernel_size, RL_stride, RL_padding, bias = RL_bias),
            nn.ReLU(ReLU_inplace),
            nn.Conv2d(RL_in, RL_out, RL_kernel_size, RL_stride, RL_padding, bias = RL_bias),
            nn.ReLU(ReLU_inplace),
            nn.Conv2d(RL_in, RL_out, RL_kernel_size, RL_stride, RL_padding, bias = RL_bias),
            nn.ReLU(ReLU_inplace),
            nn.Conv2d(RL_in, RL_out, RL_kernel_size, RL_stride, RL_padding, bias = RL_bias),
            nn.ReLU(ReLU_inplace)
        )
        self.outputLayer = nn.Sequential(
            nn.Conv2d(OL_in, OL_out, OL_kernel_size, OL_stride, OL_padding, bias = OL_bias),
        )

    def forward(self, x):
        downSample = nn.Upsample(scale_factor=1/self.scale, mode='bicubic')
        preserved = x
        output = self.inputLayer(x)
        output = self.resLayer(output)
        output = self.outputLayer(output)
        output = downSample(output)
        preserved = downSample(preserved)
        return preserved + output

