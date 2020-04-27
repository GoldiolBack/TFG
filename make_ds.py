import os
from torchvision.datasets.folder import default_loader
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


root = "C:/Users/orioe/Desktop/ETSETB/4B/TFG/Prova"
gt = "C:/Users/orioe/Desktop/ETSETB/4B/TFG/Prova/GT"
rs = "C:/Users/orioe/Desktop/ETSETB/4B/TFG/Prova/RS"
dirs = [root, gt, rs]


def make_dataset(root: str) -> list:
    """Reads a directory with data.
    Returns a dataset as a list of tuples of paired image paths: (rs_path, gt_path)
    """
    dataset = []

    # Our dir names
    rs_dir = 'RS'
    gt_dir = 'GT'

    # Get all the filenames from Rs folder
    rs_fnames = sorted(os.listdir(os.path.join(root, rs_dir)))

    # Compare file names from GT folder to file names from Rs:
    for gt_fname in sorted(os.listdir(os.path.join(root, gt_dir))):

        if gt_fname in rs_fnames:
            # if we have a match - create pair of full path to the corresponding images
            rs_path = os.path.normpath(os.path.join(root, rs_dir, gt_fname))
            gt_path = os.path.normpath(os.path.join(root, gt_dir, gt_fname))


            item = (rs_path, gt_path)
            # append to the list dataset
            dataset.append(item)
        else:
            continue

    return dataset


dataset = make_dataset(root)

# print('Our make_dataset:')
# print(*dataset, sep='\n')


class CustomVisionDataset(VisionDataset):

    def __init__(self,
                 root,
                 loader=default_loader,
                 rs_transform=None,
                 gt_transform=None):
        super().__init__(root,
                         transform=rs_transform,
                         target_transform=gt_transform)

        # Prepare dataset
        samples = make_dataset(self.root)

        self.loader = loader
        self.samples = samples
        # list of Rs images
        self.rs_samples = [s[1] for s in samples]
        # list of GT images
        self.gt_samples = [s[1] for s in samples]

    def __getitem__(self, index):
        """Returns a data sample from our dataset.
        """
        # getting our paths to images
        rs_path, gt_path = self.samples[index]

        # import each image using loader (by default it's PIL)
        rs_sample = self.loader(rs_path)
        gt_sample = self.loader(gt_path)

        # here goes tranforms if needed
        # maybe we need different tranforms for each type of image
        if self.transform is not None:
            rs_sample = self.transform(rs_sample)
        if self.target_transform is not None:
            gt_sample = self.target_transform(gt_sample)

            # now we return the right imported pair of images (tensors)
        return rs_sample, gt_sample

    def __len__(self):
            return len(self.samples)


bs=2  # batch size
transforms = ToTensor()  # we need this to convert PIL images to Tensor
shuffle = True

train = CustomVisionDataset(root, rs_transform=transforms, gt_transform=transforms)
train_loader = DataLoader(train, batch_size=bs, shuffle=shuffle)

# for i, (rs, gt) in enumerate(train_loader):
#     print(f'batch {i+1}:')
#     # some plots
#     for i in range(bs):
#         plt.figure(figsize=(10, 5))
#         plt.subplot(221)
#         plt.imshow(rs[i].squeeze().permute(1, 2, 0))
#         plt.title(f'RGB img{i+1}')
#         plt.subplot(222)
#         plt.imshow(gt[i].squeeze().permute(1, 2, 0))
#         plt.title(f'GT img{i+1}')
#         plt.show()
