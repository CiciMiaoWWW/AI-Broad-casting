import os
from tqdm import tqdm
import cv2
import numpy as np
input_path = ''
output_path = ''
if not os.path.exists(output_path):
    os.makedirs(output_path)
img_list = os.listdir(input_path)

for name in tqdm(img_list):
    img = cv2.imread(input_path+name)
    try:
        # scale = min(np.size(img,0),np.size(img,1)) / 256
        # h_new = int(np.size(img,1) / scale)
        # w_new = int(np.size(img,0) / scale)
        re_img = cv2.resize(img, (1920,1080), interpolation = cv2.INTER_CUBIC)
        cv2.imwrite(output_path+name,re_img)
        print(name)
    except:
        continue



