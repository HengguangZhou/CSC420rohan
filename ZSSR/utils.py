import torch
import numpy as np
import math
from cv2 import warpPerspective, INTER_CUBIC
import torch
from torch.nn.functional import mse_loss

def calc_psnr(img1, img2):
    img1 = torch.from_numpy(img1).float()
    img2 = torch.from_numpy(img2).float()
    mse = mse_loss(img1, img2)
    print(mse)
    return -10 * math.log10(mse) + 20 * math.log10(1.0)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count