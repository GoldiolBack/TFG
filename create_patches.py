import skimage.io as skio
from skimage.transform import resize
import sklearn.feature_extraction.image as skfi
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


date = "20190831T104021"
root_path = "images/S2A_MSIL2A_20190831T104021_N0213_R008_T31TDG_20190831T140616.SAFE"

# read the image and normalize its data
im10 = skio.imread(root_path+"/10m"+"all_bands.tiff")
im20 = skio.imread(root_path+"/20m"+"all_bands.tiff")

# create patches out of the image
i = 0
j = 0
num_patches = 800
num_final_patches = 300
size_im10 = 512
resize_im10 = 256
size_im20 = 256
resize_im20 = 128
channels10 = 4
channels20 = 6
max_pixel = np.round((im10.shape[0] - size_im10)/2).astype(dtype=np.int)
# patch_8bit = np.ndarray((num_patches, size_im10, size_im10, channels10))
patch_norm10 = np.ndarray((num_patches, size_im10, size_im10, channels10))
patch_norm20 = np.ndarray((num_patches, size_im20, size_im20, channels20))
gauss10 = np.ndarray((num_final_patches, size_im10, size_im10, channels10))
gauss20 = np.ndarray((num_final_patches, size_im20, size_im20, channels20))
rs10 = np.ndarray((num_final_patches, resize_im10, resize_im10, channels10))
rs20 = np.ndarray((num_final_patches, resize_im20, resize_im20, channels20))

patches10 = np.ndarray((num_patches, size_im10, size_im10, channels10))
patches20 = np.ndarray((num_patches, size_im20, size_im20, channels20))
patches20_target = np.ndarray((num_final_patches, size_im20, size_im20, channels20))

for i in range(num_patches):
    j10 = np.random.randint(max_pixel)*2
    j20 = np.round((j10/2)).astype(dtype=np.int)
    k10 = np.random.randint(max_pixel)*2
    k20 = np.round((k10 / 2) - 1).astype(dtype=np.int)
    patches10[i] = im10[j10:(j10+size_im10), k10:(k10+size_im10), :]
    patches20[i] = im20[j20:(j20+size_im20), k20:(k20+size_im20), :]
    print(i)


print("Finished first")

for i in range(num_patches):
    patch_norm10[i] = patches10[i]/(patches10[i].max())
    patch_norm20[i] = patches20[i]/(patches20[i].max())
    if j < num_final_patches and ((patches10[i].max() < 5000 or patches20[i].max() < 4000) or
                                  (patches10[i].max() - patches10[i].min() > 800 or
                                   patches20[i].max() - patches20[i].min() > 800)):
        gauss10[j] = gaussian_filter(patches10[i], sigma=1 / 2)
        gauss20[j] = gaussian_filter(patches20[i], sigma=1 / 2)
        rs10[j] = resize(gauss10[j], (resize_im10, resize_im10))
        rs20[j] = resize(gauss20[j], (resize_im20, resize_im20))
        patches20_target[j] = patches20[i]
        j += 1


print("Finished second")


np.save('input10_resized20.npy', rs10)
print('First finished')
np.save('input20_resized40.npy', rs20)
print('Second finished')
np.save('real20_target.npy', patches20_target)


# np.save('test_resized20.npy', rs10)
# print('First finished')
# np.save('test_resized40.npy', rs20)
# print('Second finished')
# np.save('real20_target_test.npy', patches20_target)


# fig = plt.figure()
# ax1 = fig.add_subplot(221)
# ax2 = fig.add_subplot(222)
# ax3 = fig.add_subplot(223)
# ax4 = fig.add_subplot(224)
# ax1.imshow(patch_norm10[1, :, :, 0:3])
# ax2.imshow(patch_norm20[1, :, :, 2:5])
# ax3.imshow(rs10[1, :, :, 0:3]/rs10[1, :, :, 0:3].max())
# ax4.imshow(rs20[1, :, :, 2:5]/rs20[1, :, :, 2:5].max())
# plt.show()
#
# a10 = np.loadtxt('input10_resized20.csv', delimiter=',')
# a10 = a10.reshape(100, 64, 64, 4)
# a20 = np.loadtxt('input20_resized40.csv', delimiter=',')
# a20 = a20.reshape(100, 32, 32, 6)
