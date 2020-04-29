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

imagePath = os.path.join(root_path, "GRANULE\\"+codigo_l2a+"\\IMG_DATA\\R10m\\")
band_prefix = code+"_"+date

band2 = rasterio.open(imagePath+band_prefix+'_B02_10m.jp2', driver='JP2OpenJPEG') #blue
band3 = rasterio.open(imagePath+band_prefix+'_B03_10m.jp2', driver='JP2OpenJPEG') #green
band4 = rasterio.open(imagePath+band_prefix+'_B04_10m.jp2', driver='JP2OpenJPEG') #red
band8 = rasterio.open(imagePath+band_prefix+'_B08_10m.jp2', driver='JP2OpenJPEG') #nir

print("NUMBER OF BANDs: ", band4.count)
print("Raster size: (widht= %f, height=%f) "%(band2.width, band2.height))

print(band2.driver)

# system of reference
print(band2.crs)

# type of raster
print(band2.dtypes[0])

# raster transform parameters
print(band2.transform)

# rio_plot.show(band2)

# EXPORT RASTER TIFF
raster = rasterio.open(root_path+"\\"+date+".tiff", "w", driver="Gtiff",
                       width=band2.width, height=band2.height,
                       count=4, crs=band2.crs, transform=band2.transform,
                       dtype=band2.dtypes[0])
raster.write(band4.read(1), 1)  # write band4 -red- in position 1
raster.write(band3.read(1), 2)  # write band3 -green- in position 2
raster.write(band2.read(1), 3)  # write band2 -blue- in position 3
# raster.write(band8.read(1), 4)  # write band8 -nir- in position 4
raster.close()

print(band2.width)

# rio_plot.show_hist(band2, bins=100, lw=0.0, stacked=False)
