import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
import cv2
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable
import h5py


def hr_transform(crop_size):
    return Compose([CenterCrop(crop_size),
                    ToTensor(),
                    ])

def lr_transform(crop_size, scale):
    return Compose([CenterCrop(crop_size // scale),
                    ToTensor(),
                    ])


class folderDataset(Dataset):
    def __init__(self, hr_dir, crop_size, scale, lr_dir=None, lr_transform=lr_transform, hr_transform=hr_transform):
        super(folderDataset, self).__init__()
        self.hr_images = [os.path.join(hr_dir, img) for img in os.listdir(hr_dir) if '.png' in img]
        if lr_dir:
            self.lr_images = [os.path.join(lr_dir, img) for img in os.listdir(lr_dir) if '.png' in img]
            assert len(self.hr_images) == len(self.lr_images)
        else:
            self.hr2lr_transform = Resize(crop_size // scale)
            self.lr_images = None

        self.lr_transform = lr_transform(crop_size, scale)
        self.hr_transform = hr_transform(crop_size)

    def __getitem__(self, idx):
        hr = Image.open(self.hr_images[idx]).convert()

        if self.lr_images:
            lr = Image.open(self.lr_images[idx])
        else:
            lr = self.hr2lr_transform(hr)

        if self.lr_transform:
            lr = self.lr_transform(lr)

        if self.hr_transform:
            hr = self.hr_transform(hr)

        return lr, hr

    def __len__(self):
        return len(self.hr_images)

# Adapted from https://github.com/yjn870/FSRCNN-pytorch
class h5TrainDataset(Dataset):
    def __init__(self, h5_file):
        super(h5TrainDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return np.expand_dims(f['lr'][idx] / 255., 0), np.expand_dims(f['hr'][idx] / 255., 0)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])

class h5EvalDataset(Dataset):
    def __init__(self, h5_file):
        super(h5EvalDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return np.expand_dims(f['lr'][str(idx)][:, :] / 255., 0), np.expand_dims(
                f['hr'][str(idx)][:, :] / 255., 0)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])



if __name__ == '__main__':
    test = folderDataset(hr_dir=".\\dataset\\DIV2K\\train\\HR", lr_dir=None, crop_size=255, scale=2)
    loader = DataLoader(dataset=test,
                        batch_size=1,
                        )

    for lr, hr in loader:
        print(lr.shape)
        print(hr.shape)