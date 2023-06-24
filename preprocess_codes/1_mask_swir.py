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


x=io.imread('swir_stretched.tif')
print(x.shape)

z = np.swapaxes(x, 2,0)
swir = np.swapaxes(z, 1,0)
print(swir.shape)

print(np.min(swir),np.mean(swir),np.std(swir),np.max(swir))


vis = io.imread('vis.tif')
print(vis.shape)
print(np.min(vis),np.mean(vis),np.std(vis),np.max(vis))


mask = np.zeros(vis.shape[0:2])

mask[vis[:,:,0]!=0]=255
mask[vis[:,:,1]!=0]=255
mask[vis[:,:,2]!=0]=255

plt.imshow(mask)
plt.show()




for i in range(swir.shape[2]):
    swir[:,:,i][mask==0]=0
    
plt.imshow(swir[:,:,0:3])
plt.show()

opath = 'D:\IEEE_GRSS_IADFTC_Contest2014_full_data_set\\tiffs\\'




img = np.swapaxes(swir, 2,0)
img= np.swapaxes(img, 1,2)

io.imsave(opath + 'swir_stretched_masked' + '.tif', img)