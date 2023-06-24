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

x=io.imread('D:\IEEE_GRSS_IADFTC_Contest2014_full_data_set\\tiffs\patched_masked_swir\_r02_c00.tiff')
print(x.shape)

z = np.swapaxes(x, 2,0)
img = np.swapaxes(z, 1,0)
print(img.shape)

print(np.min(img),np.mean(img),np.std(img),np.max(img))

plt.imshow(img[:,:,0:3])
plt.show()