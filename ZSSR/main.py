# reference: https://github.com/HarukiYqM/pytorch-ZSSR
import argparse
import torch
import cv2 as cv
from model import ZSSRNet
import os
import PIL
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from utils import *
from torch.autograd import Variable
import matplotlib.pyplot as plt

scale = 2
axis_list = [0, 1]
angles = [0, cv.ROTATE_90_CLOCKWISE, cv.ROTATE_180, cv.ROTATE_90_COUNTERCLOCKWISE]
angles_reverse = [0, cv.ROTATE_90_COUNTERCLOCKWISE, cv.ROTATE_180, cv.ROTATE_90_CLOCKWISE]
lr = 0.001
mse_rec, mse_steps = [], []
losses = []
crop_size = 64

def im_resize(img, scale):
    return cv.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)), interpolation=cv.INTER_CUBIC)


def get_father(img):
    min_scale = 0.7
    max_scale = 1.0
    scale = np.random.rand() * (max_scale - min_scale) + min_scale
    img = randomRotate(img)
    img = randomMirror(img)
    img = im_resize(img, scale)
    # crop
    x, y = random.randrange(img.shape[0]-crop_size), random.randrange(img.shape[1]-crop_size)
    return img[x:x+crop_size, y:y+crop_size]

def randomRotate(img):
    angle = angles[random.randrange(len(angles))]
    if angle == 0:
        return img
    else:
        return cv.rotate(img, angle)
    

def randomMirror(img):
    axis = axis_list[random.randrange(len(axis_list))]
    return cv.flip(img, axis)


def father_to_son(father):
    lr_son = im_resize(father, 1/scale)
    # add noise
    lr_son = lr_son.astype('float64')
    gaussian = np.random.normal(0, 0.0125, (lr_son.shape[0],lr_son.shape[1])) 
    lr_son[:,:,0] += gaussian
    lr_son[:,:,1] += gaussian
    lr_son[:,:,2] += gaussian
    return lr_son


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-img', type=str, required=True)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    # initialization
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    model = ZSSRNet().to(device)
    input_img = cv.imread(args.input_img)
    gt = input_img[:int(input_img.shape[0]//scale)*scale, :int(input_img.shape[1]//scale)*scale] / 255.0
    input_img = im_resize(gt, 1 / scale)

    if not os.path.exists('./output'):
        os.makedirs('./output')

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    epoch_losses = AverageMeter()

    # training process 
    i = 0
    while lr > 1e-6:
        hr_father = get_father(input_img)
        lr_son = father_to_son(hr_father)

        # train
        lr_son = im_resize(lr_son, scale)
        lr_son = torch.from_numpy(lr_son).permute(2,0,1).unsqueeze_(0).float().requires_grad_().to(device)

        hr_father = torch.from_numpy(hr_father).permute(2,0,1).unsqueeze_(0).float().to(device)

        model.train()
        output = model(lr_son)

        loss = criterion(hr_father, output)

        epoch_losses.update(loss.item(), 1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss)

        son = im_resize(input_img, 1 / scale)
        son = torch.from_numpy(im_resize(son, scale)).float().permute(2,0,1).unsqueeze_(0).to(device)
        recon = model(son).cpu().detach().permute(0,2,3,1).numpy()
        recon_loss = np.mean(np.square(input_img - np.squeeze(recon)))
        mse_rec.append(recon_loss)
        mse_steps.append(i)

        # update lr
        if i > 500 and i % 100 == 0:
            [slope, _], [[var, _], _] = np.polyfit(mse_steps[-5:], mse_rec[-5: ], 1, cov=True)
            std = np.sqrt(var)
            print(f'slope:{slope}, STD: {std}')
            print(f"Iteration {i}, Learning Rate: {lr}, Loss: {loss}")
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            if -1.5*slope < std:
                lr /= 10
                print(f"learning rate updated: {lr}")
        i += 1

    output_list = []
    for axis in axis_list:
        for angle in range(len(angles)):
            img = input_img
            if angle != 0:
                img = cv.rotate(img, angles[angle])
            if axis != -1:
                img = cv.flip(img, axis)
            img = im_resize(img, scale)
            img = torch.from_numpy(img).float().permute(2,0,1).unsqueeze_(0).to(device)

            output = model(img)
            output = output.cpu().detach().numpy()[0, ...]
            output = np.swapaxes(output, 0, 2)  
            output = np.swapaxes(output, 0, 1)
            if axis != -1:
                output = cv.flip(output, axis)
            if angle != 0:
                output = cv.rotate(output, angles[-angle])  
            output_list.append(output)
    result = np.mean(output_list, axis=0)
    print(calc_psnr(gt/255.0, result))
    result *= 255.0
    cv.imwrite('result.png', result.astype('int'))

    plt.plot(mse_rec)
    plt.savefig('losses.jpg')



    
