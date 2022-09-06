import numpy as np

def L1(img1, img2, crop_border=0):
    assert img1.shape == img2.shape, (
        f'Image shapes are different: {img1.shape}, {img2.shape}.')
    
    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, None]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, None]

    #mse_value = np.mean((img1 - img2)**2)
    l1_value = np.mean(np.abs(img1 - img2))
    return l1_value
