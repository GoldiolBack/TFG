import os
import torch
import io
from torchvision.datasets.folder import default_loader
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import numpy as np


num_final_patches = 300
num_final_patches_test = 100
resize_im10 = 256
size_im20 = 256
resize_im20 = 128
channels10 = 4
channels20 = 6

root = "C:/Users/orioe/PycharmProjects/TFG/"
file_name_HR = 'input10_resized20.npy'
file_name_LR = 'input20_resized40.npy'
file_name_target = 'real20_target.npy'

test_HR = 'test_resized20.npy'
test_LR = 'test_resized40.npy'
test_target = 'real20_target_test.npy'

# read training data from *.npy file
HR_data_np = (np.load(file_name_HR))
LR_data_np = (np.load(file_name_LR))
target_np = (np.load(file_name_target))

# read testing data from *.npy file
HR_test_np = (np.load(test_HR))
LR_test_np = (np.load(test_LR))
target_test_np = (np.load(test_target))

# transform read data in numpy to torch Tensor
HR_data = ((torch.from_numpy(HR_data_np)).permute(0, 3, 1, 2)).type(dtype=torch.float32)
LR_data = ((torch.from_numpy(LR_data_np)).permute(0, 3, 1, 2)).type(dtype=torch.float32)
target_data = ((torch.from_numpy(target_np)).permute(0, 3, 1, 2)).type(dtype=torch.float32)

HR_test = ((torch.from_numpy(HR_test_np)).permute(0, 3, 1, 2)).type(dtype=torch.float32)
LR_test = ((torch.from_numpy(LR_test_np)).permute(0, 3, 1, 2)).type(dtype=torch.float32)
target_test = ((torch.from_numpy(target_test_np)).permute(0, 3, 1, 2)).type(dtype=torch.float32)


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
train_ds, val_ds = torch.utils.data.random_split(set_ds, [270, 30])
test_ds = PatchesDataset(HR_test, LR_test, target_test)
