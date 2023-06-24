import os
m_wp_ipath = 'D:\IEEE_GRSS_IADFTC_Contest2014_full_data_set\\tiffs\_train_test\\train\\train\\vis\\' #with palm
i_wp_ipath = 'D:\IEEE_GRSS_IADFTC_Contest2014_full_data_set\\tiffs\_train_test\\train\\train\swir\\'

m_tst_path = 'D:\IEEE_GRSS_IADFTC_Contest2014_full_data_set\\tiffs\_train_test\\train\\val\\vis\\'
i_tst_path = 'D:\IEEE_GRSS_IADFTC_Contest2014_full_data_set\\tiffs\_train_test\\train\\val\swir\\'

images = os.listdir(i_wp_ipath)
masks = os.listdir(m_wp_ipath)

i=1
for img,msk in zip(images,masks):
    if i%5 ==0:
        os.rename(i_wp_ipath+img, i_tst_path+img)
        os.rename(m_wp_ipath+msk, m_tst_path+msk)

    i += 1

