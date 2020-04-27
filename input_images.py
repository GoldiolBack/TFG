from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from PIL import Image


def show_tensor(x, i):
    im = transforms.ToPILImage()(x)
    im.save(f"{i}.png")
    image = Image.open(r"5.png")
    image.show()

image0 = Image.open(r"C:/Users/orioe/Desktop/ETSETB/4B/TFG/Prova/original_test/1.png").convert("RGB")
image1 = Image.open(r"C:/Users/orioe/Desktop/ETSETB/4B/TFG/Prova/original_test/2.png").convert("RGB")
image2 = Image.open(r"C:/Users/orioe/Desktop/ETSETB/4B/TFG/Prova/original_test/3.png").convert("RGB")
image3 = Image.open(r"C:/Users/orioe/Desktop/ETSETB/4B/TFG/Prova/original_test/4.png").convert("RGB")

image0s = Image.open(r"C:/Users/orioe/Desktop/ETSETB/4B/TFG/Prova/downsample_test/1.png").convert("RGB")
image1s = Image.open(r"C:/Users/orioe/Desktop/ETSETB/4B/TFG/Prova/downsample_test/2.png").convert("RGB")
image2s = Image.open(r"C:/Users/orioe/Desktop/ETSETB/4B/TFG/Prova/downsample_test/3.png").convert("RGB")
image3s = Image.open(r"C:/Users/orioe/Desktop/ETSETB/4B/TFG/Prova/downsample_test/4.png").convert("RGB")

ups = nn.Upsample(scale_factor=2, mode='bilinear')

input128 = torch.empty([4, 3, 128, 128])
input64 = torch.empty([4, 3, 64, 64])

input128[0] = transforms.ToTensor()(image0)
input128[1] = transforms.ToTensor()(image1)
input128[2] = transforms.ToTensor()(image2)
input128[3] = transforms.ToTensor()(image3)

input64[0] = transforms.ToTensor()(image0s)
input64[1] = transforms.ToTensor()(image1s)
input64[2] = transforms.ToTensor()(image2s)
input64[3] = transforms.ToTensor()(image3s)

input64_resized = ups(input64)

training = torch.cat((input128, input64_resized))

show_tensor(training[0], 5)
show_tensor(training[1], 5)

show_tensor(training[3], 5)
show_tensor(training[4], 5)


# k = 0
# for k in range(4):
#     show_tensor(input128[k], k)

