import skimage.io as skio
from skimage.transform import resize
import sklearn.feature_extraction.image as skfi
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


date = "20190831T104021"
root_path = "C:\\Users\\orioe\\Desktop\\ETSETB\\4B\\TFG\\Images-Sentinel-2\\" \
            "S2A_MSIL2A_20190831T104021_N0213_R008_T31TDG_20190831T140616"

# read the image and normalize its data
im10 = skio.imread(root_path+"\\10m"+"all_bands.tiff")
im20 = skio.imread(root_path+"\\20m"+"all_bands.tiff")

# create patches out of the image
i = 0
num_patches = 100
size_im10 = 128
resize_im10 = 64
size_im20 = 64
resize_im20 = 32
channels10 = 4
channels20 = 6
max_pixel = np.round((im10.shape[0] - 128)/2).astype(dtype=np.int)
# patch_8bit = np.ndarray((num_patches, size_im10, size_im10, channels10))
patch_norm10 = np.ndarray((num_patches, size_im10, size_im10, channels10))
patch_norm20 = np.ndarray((num_patches, size_im20, size_im20, channels20))
gauss10 = np.ndarray((num_patches, size_im10, size_im10, channels10))
gauss20 = np.ndarray((num_patches, size_im20, size_im20, channels20))
rs10 = np.ndarray((num_patches, resize_im10, resize_im10, channels10))
rs20 = np.ndarray((num_patches, resize_im20, resize_im20, channels20))

patches10 = np.ndarray((num_patches, size_im10, size_im10, channels10))
patches20 = np.ndarray((num_patches, size_im20, size_im20, channels20))

for i in range(num_patches):
    j10 = np.random.randint(max_pixel)*2
    j20 = np.round((j10/2)).astype(dtype=np.int)
    k10 = np.random.randint(max_pixel)*2
    k20 = np.round((k10 / 2) - 1).astype(dtype=np.int)
    patches10[i] = im10[j10:(j10+size_im10), k10:(k10+size_im10), :]
    patches20[i] = im20[j20:(j20+size_im20), k20:(k20+size_im20), :]

for i in range(num_patches):
    patch_norm10[i] = patches10[i]/(patches10[i].max())
    patch_norm20[i] = patches20[i]/(patches20[i].max())
#     patch_8bit[i] = (np.round(patch_norm10[i] * 255)).astype(dtype=np.int)
    gauss10[i] = gaussian_filter(patch_norm10[i], sigma=1 / 2)
    gauss20[i] = gaussian_filter(patch_norm20[i], sigma=1 / 2)
    rs10[i] = resize(gauss10[i], (resize_im10, resize_im10))
    rs20[i] = resize(gauss20[i], (resize_im20, resize_im20))


fig = plt.figure()
ax1 = fig.add_subplot(221)
ax4 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax2 = fig.add_subplot(224)
ax1.imshow(patch_norm10[1, :, :, 1:4])
ax2.imshow(patch_norm20[1, :, :, 2:5])
ax3.imshow(rs10[1, :, :, 1:4])
ax4.imshow(rs20[1, :, :, 2:5])
plt.show()
