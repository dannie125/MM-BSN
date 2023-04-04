import cv2
import os
import numpy as np
from scipy import io

png_dir = r'output/SIDD_MMBSN_o_a45'

image_all = []
list_dir_ = os.listdir(png_dir)
dir_len = len(list_dir_)
list_dir = sorted(list_dir_)
for i in range(0, dir_len, 32):
    image_full = []
    for image_name_id in range(i, i+32):
        image_0 = cv2.imread(os.path.join(png_dir, list_dir[image_name_id]))
        image_0 = cv2.cvtColor(image_0, cv2.COLOR_RGB2BGR)
        # cv2.imshow('image_0', image_0)
        # cv2.waitKey(0)
        image_full.append(image_0)
    image_array = np.array(image_full)
    image_all.append(image_array)

image = np.array(image_all)
print(image.shape)
io.savemat('SubmitSrgb.mat', {'data':image})



