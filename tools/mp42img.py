import os 
from glob import glob
from tqdm import tqdm 
import imageio
import cv2
import numpy as np


reader_list = []

with open('../txt_folder/A.txt', 'r') as f: 
    lines = f.readlines()
    for i in lines:
        reader_list.append(i.split(' ')[0])
# print(reader_list)

root_dir = '/mnt/ljk/liuwei/data/'
output_path = root_dir+'Mahler/'
if not os.path.exists(output_path):
    os.makedirs(output_path)
vid = 'Mahler.mp4'
reader = imageio.get_reader(os.path.join(root_dir, vid))
fps = reader.get_meta_data()['fps']

num = 0
for im in reader:
    if str(num) in reader_list:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        cv2.imwrite(output_path+str(num).zfill(6)+'.png',im)
    num += 1