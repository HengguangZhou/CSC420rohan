import argparse
import os

import torch
from PIL import Image
import numpy as np
from model import FSRCNN, FLRCNN, ESPCN, DESPCN

from utils import compute_PSNR, preprocess, convert_ycbcr_to_rgb, convert_rgb_to_y, convert_rgb_to_ycbcr

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, choices=['h5', 'folder'], required=True)
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--sr_weights', type=str, required=True)
    parser.add_argument('--sr_module', type=str, choices=['FSRCNN', 'ESPCN'], required=True)
    parser.add_argument('--lr_weights', type=str, required=True)
    parser.add_argument('--lr_module', type=str, choices=['FLRCNN', 'DESPCN'], required=True)
    parser.add_argument("--scale", type=int, default=2)

    opts = parser.parse_args()

    if not os.path.exists("results"):
        os.mkdir("results")

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    if opts.sr_module == "FSRCNN":
        sr_module = FSRCNN(scale=opts.scale)
    elif opts.sr_module == "ESPCN":
        sr_module = ESPCN(scale=opts.scale)
    else:
        sr_module = FSRCNN(scale=opts.scale)

    if opts.lr_module == "FLRCNN":
        lr_module = FLRCNN(scale=opts.scale)
    elif opts.sr_module == "DESPCN":
        sr_module = DESPCN(scale=opts.scale)
    else:
        lr_module = FLRCNN(scale=opts.scale)

    sr_module = sr_module.to(device)
    lr_module = lr_module.to(device)
    sr_module.eval()
    lr_module.eval()

    sr_module.load_state_dict(torch.load(opts.sr_weights))
    lr_module.load_state_dict(torch.load(opts.lr_weights))
    image = Image.open(opts.image).convert('RGB')

    image_width = (image.width // opts.scale) * opts.scale
    image_height = (image.height // opts.scale) * opts.scale
    if opts.data_type == 'h5':
        # h5 and ycbcr processing from https://github.com/yjn870/FSRCNN-pytorch
        hr = image.resize((image_width, image_height), resample=Image.BICUBIC)
        lr = hr.resize((hr.width // opts.scale, hr.height // opts.scale), resample=Image.BICUBIC)
        bicubic = lr.resize((lr.width * opts.scale, lr.height * opts.scale), resample=Image.BICUBIC)

        lr, ycbcr_lr = preprocess(lr, device)
        hr, _ = preprocess(hr, device)
        bicubic_hr, ycbcr = preprocess(bicubic, device)
        print(f"bicubic PSNR: {compute_PSNR(bicubic_hr, hr)}")

        with torch.no_grad():
            sr = sr_module(lr).clamp(0.0, 1.0)
            lsr = lr_module(sr).clamp(0.0, 1.0)

        print(f"SR Module PSNR: {compute_PSNR(hr, sr)}")
        print(f"LR Module PSNR: {compute_PSNR(lr, lsr)}")

        sr = sr.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
        sr = np.array([sr, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
        output = np.clip(convert_ycbcr_to_rgb(sr), 0.0, 255.0).astype(np.uint8)
        output = Image.fromarray(output)
        output.save(opts.image.replace('.', '_fsrcnn_upscale_x{}.'.format(opts.scale)))

        lr = lr.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
        lr = np.array([lr, ycbcr_lr[..., 1], ycbcr_lr[..., 2]]).transpose([1, 2, 0])
        lr = np.clip(convert_ycbcr_to_rgb(lr), 0.0, 255.0).astype(np.uint8)
        lr = Image.fromarray(lr)
        lr.save(opts.image.replace('.', '_fsrcnn_downscale_x{}.'.format(opts.scale)))

    else: #todo
        hr = Image.open(opts.image_file).convert('RGB')
        lr = hr.resize((hr.width // opts.scale, hr.height // opts.scale), resample=Image.BICUBIC)
