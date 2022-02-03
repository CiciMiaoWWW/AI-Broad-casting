#!/usr/bin/env python

import numpy as np
import os.path
import shutil
from scipy.io.matlab.mio import savemat, loadmat
import network
import argparse
import os
import time
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from skimage import color
import utils
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--netroot', type = str, default = 'checkpoints/xixi_recon.pth', help = 'model root')
parser.add_argument('--use_ensemble', type = bool, default = True, help = 'using True for the best performance and please change this to False for runtime testing')
parser.add_argument('--saveroot', type = str, default = './test_results', help = 'result images saveroot')
parser.add_argument('--test_input_folder', type = str, default = './test_input_folder', help = 'color image baseroot')
opt = parser.parse_args()

if not os.path.exists(opt.saveroot):
    os.makedirs(opt.saveroot)
if not os.path.exists(opt.test_input_folder):
    os.makedirs(opt.test_input_folder)
test_input_list = os.listdir(opt.test_input_folder)

network = torch.load(opt.netroot).cuda()

for name in test_input_list:
    img = Image.open(opt.test_input_folder+'/'+name).convert('RGB')
    img = transforms.Resize((256,256), Image.BICUBIC)(img)
    img = np.array(img).astype(np.float32)
    img = img / 255
    img = img[np.newaxis,:,:,:]
    img = torch.from_numpy(img.transpose(0,3, 1, 2).astype(np.float32)).contiguous().cuda()
    
    # single image —— using this code to test runtime
    time_start=time.time()
    output = network(img)    
    time_end=time.time()
    
    # Recover normalization
    utils.save_sample(output, opt.saveroot,name, pixel_max_cnt = 255)
    print(name,' saved | time cost:',time_end-time_start)
