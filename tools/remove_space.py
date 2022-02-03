import os
img_folder = ''
img_list = os.listdir(img_folder)
for name in img_list:
    name_new = name.replace(' ','')
    os.rename(img_folder+name, img_folder+name_new)