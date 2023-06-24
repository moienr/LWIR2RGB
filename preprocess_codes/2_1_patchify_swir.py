from skimage import io
import numpy as np
import matplotlib.pyplot as plt

from unittest.mock import patch
import numpy as np
import matplotlib.pyplot as plt
from patchify import patchify
from skimage import io
import os
import tifffile

x=io.imread('swir_stretched_masked.tif')
print(x.shape)

z = np.swapaxes(x, 2,0)
img = np.swapaxes(z, 1,0)
print(img.shape)

print(np.min(img),np.mean(img),np.std(img),np.max(img))


# def stretch(img,sigma =3,plot_hist=False):
#     stretched = np.zeros(img.shape) 
#     for i in range(img.shape[2]):  #looping through the bands
#         band = np.zeros(img.shape[0:2]) 
#         band = img[:,:,i] # copiying each band into the variable `band`
#         if np.min(band)<0: # if the min is less that zero, first we add min to all pixels so min becomes 0
#             band = band + np.abs(np.min(band)) 
#         band = band / np.max(band)
#         band = band * 255 # convertaning values to 0-255 range
#         if plot_hist:
#             plt.hist(band.ravel(), bins=256) #calculating histogram
#             plt.show()
#         # plt.imshow(band)
#         # plt.show()
#         std = np.std(band)
#         mean = np.mean(band)
#         max = mean+(sigma*std)
#         min = mean-(sigma*std)
#         band = (band-min)/(max-min)
#         band = band * 255
#         # this streching cuases the values less than `mean-simga*std` to become negative
#         # and values greater than `mean+simga*std` to become more than 255
#         # so we clip the values ls 0 and gt 255
#         band[band>255]=255  
#         band[band<0]=0
#         print('band',i,np.min(band),np.mean(band),np.std(band),np.max(band))
#         if plot_hist:
#             plt.hist(band.ravel(), bins=256) #calculating histogram
#             plt.show()
#         stretched[:,:,i] = band
        
        
#     stretched = stretched.astype('int')
#     return stretched

# st = stretch(z,sigma=3,plot_hist=False)
# plt.imshow(st[:,:,0:3],vmin=0,vmax=255)
# plt.show()


  
# img = st


opath = 'D:\IEEE_GRSS_IADFTC_Contest2014_full_data_set\\tiffs\patched_masked_swir\\'

patches = patchify(img,(64,64,7),step=64-8)
print(patches.shape)
patches = np.squeeze(patches)
print(patches.shape)
    

for i in range(patches.shape[0]):
    for j in range(patches.shape[1]):
        patch = patches[i,j,:,:,:]
        print(patch.shape)
        patch = np.swapaxes(patch, 2,0)
        patch= np.swapaxes(patch, 1,2)
        
        io.imsave(opath + '_r'+ str(i).zfill(2) + '_c' + str(j).zfill(2) + '.tiff', patch)


# io.imsave(opath + file.split('.')[0] + '.jpg',x)

    