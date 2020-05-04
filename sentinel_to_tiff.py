import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import earthpy.plot as ep

from PIL import Image
import skimage.io as skio

import rasterio
from rasterio import plot as rio_plot
import os

# fuente: https://www.hatarilabs.com/ih-en/sentinel2-images-explotarion-and-processing-with-python-and-rasterio

date = "20190831T104021"
code = "T31TDG"

root_path = "C:\\Users\\orioe\\Desktop\\ETSETB\\4B\\TFG\\Images-Sentinel-2\\" \
            "S2A_MSIL2A_20190831T104021_N0213_R008_T31TDG_20190831T140616"

codigo_l2a= "L2A_T31TDG_A021884_20190831T104349"

imagePath = os.path.join(root_path, "GRANULE\\"+codigo_l2a+"\\IMG_DATA\\R20m\\")
band_prefix = code+"_"+date

# band2 = rasterio.open(imagePath+band_prefix+'_B02_20m.jp2', driver='JP2OpenJPEG')  #blue
# band3 = rasterio.open(imagePath+band_prefix+'_B03_20m.jp2', driver='JP2OpenJPEG')  #green
# band4 = rasterio.open(imagePath+band_prefix+'_B04_20m.jp2', driver='JP2OpenJPEG')  #red
# band8 = rasterio.open(imagePath+band_prefix+'_B08_20m.jp2', driver='JP2OpenJPEG')  #nir
band5 = rasterio.open(imagePath+band_prefix+'_B05_20m.jp2', driver='JP2OpenJPEG')  #nir
band6 = rasterio.open(imagePath+band_prefix+'_B06_20m.jp2', driver='JP2OpenJPEG')  #nir
band7 = rasterio.open(imagePath+band_prefix+'_B07_20m.jp2', driver='JP2OpenJPEG')  #nir
band8a = rasterio.open(imagePath+band_prefix+'_B8A_20m.jp2', driver='JP2OpenJPEG')  #nir
band11 = rasterio.open(imagePath+band_prefix+'_B11_20m.jp2', driver='JP2OpenJPEG')  #nir
band12 = rasterio.open(imagePath+band_prefix+'_B12_20m.jp2', driver='JP2OpenJPEG')  #nir

print("NUMBER OF BANDs: ", band12.count)
print("Raster size: (widht= %f, height=%f) "%(band11.width, band11.height))

print(band11.driver)

# system of reference
print(band12.crs)

# type of raster
print(band11.dtypes[0])

# raster transform parameters
print(band11.transform)

# rio_plot.show(band2)

# EXPORT RASTER TIFF
raster = rasterio.open(root_path+"\\20m"+"all_bands.tiff", "w", driver="Gtiff",
                       width=band5.width, height=band5.height,
                       count=6, crs=band5.crs, transform=band5.transform,
                       dtype=band5.dtypes[0])
# raster.write(band4.read(1), 1)  # write band4 -red- in position 1
# raster.write(band3.read(1), 2)  # write band3 -green- in position 2
# raster.write(band2.read(1), 3)  # write band2 -blue- in position 3
# raster.write(band8.read(1), 4)  # write band8 -nir- in position 4
raster.write(band5.read(1), 1)  # write band5 -- in position 5
raster.write(band6.read(1), 2)  # write band6 -- in position 6
raster.write(band7.read(1), 3)  # write band7 -- in position 7
raster.write(band8a.read(1), 4)  # write band8a -- in position 8
raster.write(band11.read(1), 5)  # write band11 -- in position 9
raster.write(band12.read(1), 6)  # write band12 -- in position 10

raster.close()

print(band5.width)

# rio_plot.show_hist(band2, bins=100, lw=0.0, stacked=False)
