import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import os

m_ipath = 'D:\IEEE_GRSS_IADFTC_Contest2014_full_data_set\\tiffs\_train_test\\train\\augmented\\vis\\' #with palm
i_ipath = 'D:\IEEE_GRSS_IADFTC_Contest2014_full_data_set\\tiffs\_train_test\\train\\augmented\\swir\\'



images = os.listdir(i_ipath)
masks = os.listdir(m_ipath)

i=1
for image,mask in zip(images,masks):
    # img = io.imread(i_ipath+image)
    msk = io.imread(m_ipath+mask)
    
    # msk[msk<200]=0 
    
    if (np.count_nonzero(msk[:,:,0])/64**2) <= 0.3:
        os.remove(i_ipath+image)
        os.remove(m_ipath+mask)
        
    


