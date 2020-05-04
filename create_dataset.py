import os
import torch
import io
from torchvision.datasets.folder import default_loader
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import numpy as np


num_final_patches = 100
num_final_patches_test = 50
resize_im10 = 64
size_im20 = 64
resize_im20 = 32
channels10 = 4
channels20 = 6

root = "C:/Users/orioe/PycharmProjects/TFG/"
file_name_HR = 'input10_resized20.csv'
file_name_LR = 'input20_resized40.csv'
file_name_target = 'real20_target.csv'

test_HR = 'test_resized20.csv'
test_LR = 'test_resized40.csv'
test_target = 'real20_target_test.csv'

# read training data from *.csv file
HR_data_np = (np.loadtxt(file_name_HR, delimiter=',')).reshape((num_final_patches, resize_im10, resize_im10, channels10))
LR_data_np = (np.loadtxt(file_name_LR, delimiter=',')).reshape((num_final_patches, resize_im20, resize_im20, channels20))
target_np = (np.loadtxt(file_name_target, delimiter=',')).reshape((num_final_patches, size_im20, size_im20, channels20))

# read testing data from *.csv file
HR_test_np = (np.loadtxt(test_HR, delimiter=',')).reshape((num_final_patches_test, resize_im10, resize_im10, channels10))
LR_test_np = (np.loadtxt(test_LR, delimiter=',')).reshape((num_final_patches_test, resize_im20, resize_im20, channels20))
target_test_np = (np.loadtxt(test_target, delimiter=',')).reshape((num_final_patches_test, size_im20, size_im20, channels20))

# transform read data in numpy to torch Tensor
HR_data = (torch.from_numpy(HR_data_np)).permute(0, 3, 1, 2)
LR_data = (torch.from_numpy(LR_data_np)).permute(0, 3, 1, 2)
target_data = (torch.from_numpy(target_np)).permute(0, 3, 1, 2)

HR_test = (torch.from_numpy(HR_test_np)).permute(0, 3, 1, 2)
LR_test = (torch.from_numpy(LR_test_np)).permute(0, 3, 1, 2)
target_test = (torch.from_numpy(target_test_np)).permute(0, 3, 1, 2)


class PatchesDataset(Dataset):
    """Patches dataset."""

    def __init__(self, hr, lr, target, transform=None):
        """
        Args:
            HRdata: images of high resolution
            LRdata: images of low resolution
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.hr = hr
        self.lr = lr
        self.target = target
        self.transform = transform

    def __len__(self):
        return len(self.hr)

    def __getitem__(self, idx):

        hr_data = self.hr[idx]
        lr_data = self.lr[idx]
        t_data = self.target[idx]

        if self.transform is not None:
            hr_data = self.transform(self.hr)
            lr_data = self.transform(self.lr)
            t_data = self.transform(self.target)

        return hr_data, lr_data, t_data


set_ds = PatchesDataset(HR_data, LR_data, target_data)
train_ds, val_ds = torch.utils.data.random_split(set_ds, [90, 10])
test_ds = PatchesDataset(HR_test, LR_test, target_test)
