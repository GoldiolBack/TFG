import os
from torchvision.datasets.folder import default_loader
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


root = "C:/Users/orioe/Desktop/ETSETB/4B/TFG/Prova"
gt_test = "C:/Users/orioe/Desktop/ETSETB/4B/TFG/Prova/GT_test"
rs_test = "C:/Users/orioe/Desktop/ETSETB/4B/TFG/Prova/RS_test"
dirs_test = [root, gt_test, rs_test]


def make_dataset(root: str) -> list:
    """Reads a directory with data.
    Returns a dataset as a list of tuples of paired image paths: (rs_test_path, gt_test_path)
    """
    dataset = []

    # Our dir names
    rs_test_dir = 'RS_test'
    gt_test_dir = 'GT_test'

    # Get all the filenames from Rs folder
    rs_test_fnames = sorted(os.listdir(os.path.join(root, rs_test_dir)))

    # Compare file names from GT_test folder to file names from Rs:
    for gt_test_fname in sorted(os.listdir(os.path.join(root, gt_test_dir))):

        if gt_test_fname in rs_test_fnames:
            # if we have a match - create pair of full path to the corresponding images
            rs_test_path = os.path.normpath(os.path.join(root, rs_test_dir, gt_test_fname))
            gt_test_path = os.path.normpath(os.path.join(root, gt_test_dir, gt_test_fname))


            item = (rs_test_path, gt_test_path)
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
                 rs_test_transform=None,
                 gt_test_transform=None):
        super().__init__(root,
                         transform=rs_test_transform,
                         target_transform=gt_test_transform)

        # Prepare dataset
        samples = make_dataset(self.root)

        self.loader = loader
        self.samples = samples
        # list of Rs images
        self.rs_test_samples = [s[1] for s in samples]
        # list of GT_test images
        self.gt_test_samples = [s[1] for s in samples]

    def __getitem__(self, index):
        """Returns a data sample from our dataset.
        """
        # getting our paths to images
        rs_test_path, gt_test_path = self.samples[index]

        # import each image using loader (by default it's PIL)
        rs_test_sample = self.loader(rs_test_path)
        gt_test_sample = self.loader(gt_test_path)

        # here goes tranforms if needed
        # maybe we need different tranforms for each type of image
        if self.transform is not None:
            rs_test_sample = self.transform(rs_test_sample)
        if self.target_transform is not None:
            gt_test_sample = self.target_transform(gt_test_sample)

            # now we return the right imported pair of images (tensors_test)
        return rs_test_sample, gt_test_sample

    def __len__(self):
            return len(self.samples)


bs=2  # batch size
transforms = ToTensor()  # we need this to convert PIL images to Tensor
shuffle = True

test = CustomVisionDataset(root, rs_test_transform=transforms, gt_test_transform=transforms)
test_loader = DataLoader(test, batch_size=bs, shuffle=shuffle)

# for i, (rs_test, gt_test) in enumerate(test_loader):
#     print(f'batch {i+1}:')
#     # some plots
#     for i in range(bs):
#         plt.figure(figsize=(10, 5))
#         plt.subplot(221)
#         plt.imshow(rs_test[i].squeeze().permute(1, 2, 0))
#         plt.title(f'RGB img{i+1}')
#         plt.subplot(222)
#         plt.imshow(gt_test[i].squeeze().permute(1, 2, 0))
#         plt.title(f'GT_test img{i+1}')
#         plt.show()
