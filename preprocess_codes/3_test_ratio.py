
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import os

path = 'D:\IEEE_GRSS_IADFTC_Contest2014_full_data_set\\tiffs\patched_vis\\'

images = os.listdir(path)
images.sort()


for image in images:
    img=io.imread(path+image)
    # img[img<200]=0 
    
    band = img[:,:,0]
    # print(np.count_nonzero(band))
    print(image,' : ',np.count_nonzero(band)/64**2)