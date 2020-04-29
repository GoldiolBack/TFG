import skimage.io as skio
from skimage import exposure
import sklearn.feature_extraction.image as skfi
import numpy as np


date = "20190831T104021"
root_path = "C:\\Users\\orioe\\Desktop\\ETSETB\\4B\\TFG\\Images-Sentinel-2\\" \
            "S2A_MSIL2A_20190831T104021_N0213_R008_T31TDG_20190831T140616"

# read the image and normalize its data
im = skio.imread(root_path+"\\"+date+".tiff")
maximum = im.max()
image = im / maximum

# create patches out of the image
i = 0
num_patches = 100
size_im = 128
channels = 3
patch_eq = np.ndarray((num_patches, size_im, size_im, channels))
patches = skfi.extract_patches_2d(image, (size_im, size_im), num_patches)
for i in range(num_patches):
    patch_eq[i] = exposure.equalize_adapthist(patches[i])
