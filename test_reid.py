#!/usr/bin/env python

import numpy as np
import os.path
import shutil
from scipy.io.matlab.mio import savemat, loadmat
from borrowed_networks.reid_model import ft_net
import argparse
import os
from torch.autograd import Variable
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import pandas as pd
import utils
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--netroot', type = str, default = 'net_last.pth', help = 'model root')
parser.add_argument('--use_ensemble', type = bool, default = True, help = 'using True for the best performance and please change this to False for runtime testing')
parser.add_argument('--saveroot', type = str, default = './test_results/', help = 'result images saveroot')
parser.add_argument('--gallery_folder', type = str, default = '/mnt/bd/lwlq-workshop/Person_reID_baseline_pytorch/basic_codes/runs/detect/exp/gallery', help = 'gallery image baseroot')
parser.add_argument('--query_folder', type = str, default = '/mnt/bd/lwlq-workshop/Person_reID_baseline_pytorch/basic_codes/runs/detect/exp/crops', help = 'query image baseroot')
parser.add_argument('--yolo_txt', type = str, default = '/mnt/bd/lwlq-workshop/Person_reID_baseline_pytorch/basic_codes/runs/detect/exp/labels/飞书20220129-210549.mp4.txt', help = 'yolo outputs')
parser.add_argument('--nclasses', type = int, default = 751, help = 'nclasses')
parser.add_argument('--stride', type = int, default = 2, help = 'stride')
parser.add_argument('--ibn', action='store_true', help='use ibn.' )
parser.add_argument('--linear_num', type = int, default = 512, help = 'linear_num')
parser.add_argument('--PCB', action='store_true', help='use PCB')
opt = parser.parse_args()

if not os.path.exists(opt.saveroot):
    os.makedirs(opt.saveroot)

gallery_list = os.listdir(opt.gallery_folder)
query_list = os.listdir(opt.query_folder)

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        #filename = path.split('/')[-1]
        filename = os.path.basename(path)
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels

model = ft_net(opt.nclasses, stride = opt.stride, ibn = opt.ibn, linear_num=opt.linear_num)
model.load_state_dict(torch.load(opt.netroot))
model.classifier.classifier = nn.Sequential()
model = model.eval()
model = model.cuda()


gallery_features = torch.FloatTensor()
for name in gallery_list:
    data_transforms = transforms.Compose([
        transforms.Resize((256, 128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    img = Image.open(opt.gallery_folder+'/'+name).convert('RGB')
    img = data_transforms(img)
    img = img.float().unsqueeze(0)

    ff = torch.FloatTensor(1,opt.linear_num).zero_().cuda()
    for i in range(2):
        if(i==1):
            img = fliplr(img)
        input_img = Variable(img.cuda())
        outputs = model(input_img) 
        ff += outputs
    # norm feature
    fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
    ff = ff.div(fnorm.expand_as(ff))
    gallery_features = torch.cat((gallery_features,ff.data.cpu()), 0)
    # print(gallery_features.shape)

query_list = []
reader_list = []
with open(opt.yolo_txt, 'r') as f: 
    lines = f.readlines()
    for i in lines:
        i_ = i.strip()
        reader_list.append(i_)
        query_list.append(i_.split(' ')[-1])

df = pd.DataFrame(columns=['frameid','x1','y1','x2','y2','conf','path','actor_id'])
with open(opt.yolo_txt[:-8]+'_labeled.txt', 'w') as f:
    num = 0
    for (name,info) in zip(query_list,reader_list):
        data_transforms = transforms.Compose([
            transforms.Resize((256, 128), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        img = Image.open(name).convert('RGB')
        img = data_transforms(img)
        img = img.float().unsqueeze(0)

        ff = torch.FloatTensor(1,opt.linear_num).zero_().cuda()
        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            outputs = model(input_img) 
            ff += outputs
        # norm feature
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        # print(gallery_features.shape)
        # print(ff.shape)
        score = np.dot(gallery_features,np.transpose(ff.data.cpu().numpy()))
        # print(score)
        # predict index
        score = score.flatten()
        # print(score)
        maxindex = np.argmax(score)
        # print(maxindex)
        info_list = info.split(' ')
        info_list.append(maxindex)
        # print(info_list)
        df.loc[num] = info_list
        f.write('%s %s\n'%(info,maxindex))
        num+=1

df.to_pickle(opt.yolo_txt[:-8]+'_labeled.pkl')

# df=pd.read_pickle(opt.yolo_txt[:-8]+'_labeled.pkl')
# print(df)