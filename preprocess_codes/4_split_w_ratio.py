from unittest.mock import patch
import numpy as np
import matplotlib.pyplot as plt
from patchify import patchify
from skimage import io
import os

v_path = 'D:\IEEE_GRSS_IADFTC_Contest2014_full_data_set\\tiffs\patched_vis\\'
s_path = 'D:\IEEE_GRSS_IADFTC_Contest2014_full_data_set\\tiffs\patched_masked_swir\\'

v_wp_opath = 'D:\IEEE_GRSS_IADFTC_Contest2014_full_data_set\\tiffs\_train_test\\train\\vis\\' #with palm
s_wp_opath = 'D:\IEEE_GRSS_IADFTC_Contest2014_full_data_set\\tiffs\_train_test\\train\swir\\'

v_wop_opath = 'D:\IEEE_GRSS_IADFTC_Contest2014_full_data_set\\tiffs\_train_test\\test\\vis\\' #witouth palm
s_wop_opath = 'D:\IEEE_GRSS_IADFTC_Contest2014_full_data_set\\tiffs\_train_test\\test\swir\\'

swir = os.listdir(s_path)
swir.sort()

vis = os.listdir(v_path)
vis.sort()

for sw,vi in zip(swir,vis):
    s = io.imread(s_path+sw)
    v = io.imread(v_path+vi)
    band = v[:,:,0]
    if np.count_nonzero(band)/64**2>=0.1:
        io.imsave(s_wp_opath+sw,s)
        io.imsave(v_wp_opath+vi,v)
    else:
        io.imsave(s_wop_opath+sw,s)
        io.imsave(v_wop_opath+vi,v)

