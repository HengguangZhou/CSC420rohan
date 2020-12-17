import argparse
import os

import torch
from torch import nn
import torch.optim as optim
from data import folderDataset, h5TrainDataset, h5EvalDataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from model import FSRCNN, FLRCNN, ESPCN, DESPCN

from utils import compute_PSNR

if __name__ == "__main__":
    writer = SummaryWriter()
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, choices=['h5', 'folder'], required=True)
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--eval_data', type=str, required=True)
    parser.add_argument('--weights_dir', type=str, default="weights/")
    parser.add_argument('--checkpoints_dir', type=str, default="checkpoints/")
    parser.add_argument('--crop_size', type=int, default=255)
    parser.add_argument('--sr_module', type=str, default="FSRCNN")
    parser.add_argument('--lr_module', type=str, default="FLRCNN")
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--label_percents", type=float, default=0.3)
    parser.add_argument('--num_workers', type=int, default=0)

    opts = parser.parse_args()

    if not os.path.exists(opts.weights_dir):
        os.mkdir(opts.weights_dir)

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    torch.manual_seed(42)

    if opts.sr_module == "FSRCNN":
        sr_module = FSRCNN(scale=opts.scale).to(device)
    elif opts.sr_module == "ESPCN":
        sr_module = ESPCN(scale=opts.scale).to(device)
    else:
        sr_module = FSRCNN(scale=opts.scale).to(device)

    if opts.lr_module == "FLRCNN":
        lr_module = FLRCNN(scale=opts.scale).to(device)
    elif opts.sr_module == "DESPCN":
        sr_module = DESPCN(scale=opts.scale).to(device)
    else:
        lr_module = FLRCNN(scale=opts.scale).to(device)

    criterion = nn.MSELoss()

    optimizer = optim.Adam([
        {'params': sr_module.parameters()},
        {'params': lr_module.parameters()}
    ], lr=1e-6)

    if opts.data_type == 'h5':
        train_data = h5TrainDataset(h5_file=opts.train_data)
        eval_data = h5EvalDataset(h5_file=opts.eval_data)
    else:
        eval_data = folderDataset(hr_dir=opts.eval_data + 'HR/',
                                  lr_dir=opts.eval_data + 'LR/',
                                  crop_size=opts.crop_size,
                                  scale=opts.scale
                                  )
        train_data = folderDataset(hr_dir=opts.train_data + 'HR/',
                                   lr_dir=opts.train_data + 'LR/',
                                   crop_size=opts.crop_size,
                                   scale=opts.scale
                                   )

    labeled, unlabeled = random_split(dataset=train_data,
                                      lengths=[int(opts.label_percents * len(train_data)),
                                               int(np.ceil((1 - opts.label_percents) * len(train_data)))])
    labeled_loader = DataLoader(dataset=labeled,
                                batch_size=opts.batch_size,
                                shuffle=True,
                                num_workers=opts.num_workers,
                                )

    unlabeled_loader = DataLoader(dataset=unlabeled,
                                batch_size=opts.batch_size,
                                shuffle=True,
                                num_workers=opts.num_workers,
                                )


    eval_loader = DataLoader(dataset=eval_data,
                                batch_size=1,
                                shuffle=True,
                                num_workers=opts.num_workers,
                                )

    best_psnr = 0
    for epoch in range(opts.num_epochs):
        sr_module.train()
        lr_module.train()

        # Train on unlabeled data
        with tqdm(total=(len(labeled) - len(labeled) % opts.batch_size)) as t:
            t.set_description(f'labeled epoch: {epoch}/{opts.num_epochs - 1}')
            labeled_losses = 0
            for idx, data in enumerate(labeled_loader):
                lr, hr = data

                lr = lr.to(device)
                hr = hr.to(device)

                sr = sr_module(lr)
                lhr = lr_module(hr)
                loss_lr = criterion(lhr, lr)
                loss_sr = criterion(sr, hr)
                labeled_losses += loss_sr.item()
                optimizer.zero_grad()
                loss_lr.backward(retain_graph=True)
                loss_sr.backward()
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(labeled_losses / (idx + 1)))
                t.update(len(lr))

            writer.add_scalar('Train/labeled_train_loss', labeled_losses / len(labeled_loader), epoch)

        # Train on labeled data
        with tqdm(total=(len(unlabeled) - len(unlabeled) % opts.batch_size)) as t:
            unlabeled_losses = 0
            t.set_description(f'unlabeled epoch: {epoch}/{opts.num_epochs - 1}')
            for idx, data in enumerate(unlabeled_loader):
                lr, _ = data

                lr = lr.to(device)

                sr = sr_module(lr)
                lsr = lr_module(sr)

                loss_sr = criterion(lsr, lr)
                unlabeled_losses += loss_sr.item()
                optimizer.zero_grad()
                loss_sr.backward()
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(unlabeled_losses / (idx + 1)))
                t.update(len(lr))

            writer.add_scalar('Train/unlabeled_train_loss', unlabeled_losses / len(unlabeled_loader), epoch)

        torch.save(sr_module.state_dict(), os.path.join(opts.weights_dir, f'epoch_sr_module_{epoch}.pth'))
        torch.save(lr_module.state_dict(), os.path.join(opts.weights_dir, f'epoch_lr_module_{epoch}.pth'))

        # Validation
        sr_module.eval()
        lr_module.eval()
        epoch_sr_psnr = 0
        epoch_lr_psnr = 0
        # print(len(eval_loader))
        for idx, data in enumerate(eval_loader):
            lr, hr = data

            lr = lr.to(device)
            hr = hr.to(device)
            with torch.no_grad():
                sr = sr_module(lr).clamp(0.0, 1.0)
                lhr = lr_module(hr).clamp(0.0, 1.0)

            epoch_sr_psnr += compute_PSNR(sr, hr)
            epoch_lr_psnr += compute_PSNR(lhr, lr)

        epoch_sr_psnr /= len(eval_loader)
        epoch_lr_psnr /= len(eval_loader)
        writer.add_scalar('Val/train_upscaler PSNR', epoch_sr_psnr, epoch)
        writer.add_scalar('Val/train_downscale PSNR', epoch_lr_psnr, epoch)
        print(f'eval sr psnr: {epoch_sr_psnr}')
        print(f'eval lr psnr: {epoch_lr_psnr}')

        if epoch_sr_psnr / len(eval_loader) > best_psnr:
            best_psnr = epoch_sr_psnr
            torch.save(sr_module.state_dict(), os.path.join(opts.weights_dir, 'best.pth'))






