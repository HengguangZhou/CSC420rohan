import math
from torch import nn
import torch

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