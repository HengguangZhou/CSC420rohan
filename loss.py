import torch
from torch import nn
from torchvision.models.vgg import vgg16

# Adapted from https://github.com/leftthomas/SRGAN
class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]

class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        self.loss_generator = nn.Sequential(*list(vgg.features)[:35]).eval()
        for param in self.loss_generator.parameters():
            param.requires_grad = False

        self.distance = nn.MSELoss()

    def forawrd(self, input, label):
        return self.distance(self.loss_generator(input), self.loss_generator(label))