import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
img_folder = ''
img_list = os.listdir(img_folder)
i=422
for name in img_list:
    # name_new = name.split('-')[0]+'.png'
    i+=1
    name_new = str(i)+'.png'
    try:
        os.rename(img_folder+name, img_folder+name_new)
    except:
        continue


# for name in tqdm(img_list):
#     img = cv2.imread(img_folder+name)
#     img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#     cv2.imwrite('img_folder'+name,img.astype(np.uint8))