import time
import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import dataset
import utils
import os
import scipy.io
from utils import load_dict
import cv2
from torchvision import transforms
import pickle

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def _tensor_size(t):
    return t.size()[1]*t.size()[2]*t.size()[3]


def MyDNN(opt):
    # ----------------------------------------
    #       Network training parameters
    # ----------------------------------------

    # cudnn benchmark
    cudnn.benchmark = opt.cudnn_benchmark

    # configurations
    save_folder = os.path.join(opt.save_path, opt.task)
    sample_folder = os.path.join(opt.sample_path, opt.task)
    checkpoint_folder = os.path.join(opt.checkpoint_path, opt.task)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if not os.path.exists(sample_folder):
        os.makedirs(sample_folder)
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)
    # Loss functions
    criterion_L1 = torch.nn.MSELoss().cuda()
    
    # Initialize Generator
    print('creating generator')
    generator = utils.create_LXMERT(opt)
    use_checkpoint = False
    if use_checkpoint:
        checkpoint_path = './MyDNN1_denoise_epoch175_bs1'
        # Load a pre-trained network
        pretrained_net = torch.load(checkpoint_path + '.pth')
        load_dict(generator, pretrained_net)
        print('Generator is loaded!')
    # To device
    if opt.multi_gpu:
        generator = nn.DataParallel(generator)
        generator = generator.cuda()
    else:
        generator = generator.cuda()

    # stroke_order_model = torch.load('./11074.514046.ckpt')
    # stroke_order_decoder = stroke_order_model['decoder'].cuda().eval()
    # stroke_order_encoder = stroke_order_model['encoder'].cuda().eval()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr = opt.lr_g, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)

    # Learning rate decrease
    def adjust_learning_rate(opt, epoch, iteration, optimizer):
        #Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs
        if opt.lr_decrease_mode == 'epoch':
            lr = opt.lr_g * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
            if epoch < 200:
                lr = 0.0001
            if epoch >= 200:
                lr = 0.00005
            if epoch >= 300:
                lr = 0.00001
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        if opt.lr_decrease_mode == 'iter':
            lr = opt.lr_g * (opt.lr_decrease_factor ** (iteration // opt.lr_decrease_iter))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        return lr
    
    # Save the model if pre_train == True
    def save_model(opt, epoch, iteration, len_dataset, generator):
        """Save the model at "checkpoint_interval" and its multiple"""
        if opt.multi_gpu == True:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    if opt.save_name_mode:
                        torch.save(generator.module, opt.checkpoint_path + 'hanzi_sketch_%s_epoch%d_bs%d.pth' % (opt.task, epoch, opt.batch_size))
                        print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    if opt.save_name_mode:
                        torch.save(generator.module, opt.checkpoint_path + 'hanzi_sketch_%s_iter%d_bs%d.pth' % (opt.task, iteration, opt.batch_size))
                        print('The trained model is successfully saved at iteration %d' % (iteration))
        else:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    if opt.save_name_mode:
                        torch.save(generator, opt.checkpoint_path + 'hanzi_sketch_%s_epoch%d_bs%d.pth' % (opt.task, epoch, opt.batch_size))
                        print('The trained model is successfully saved at epoch %d' % (epoch))

            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    if opt.save_name_mode:
                        torch.save(generator, opt.checkpoint_path + 'hanzi_sketch_%s_iter%d_bs%d.pth' % (opt.task, iteration, opt.batch_size))
                        print('The trained model is successfully saved at iteration %d' % (iteration))

    # ----------------------------------------
    #             Network dataset
    # ----------------------------------------


    # Define the dataloader
    # trainset = dataset.TestDataset(opt)
    dict_path='Dic_stroke_order.pkl'
    namedict=open(dict_path,'rb')
    namedict=pickle.load(namedict)
    trainset = dataset.LXMERTDataset(opt,namedict)
    print('The overall number of training images:', len(trainset))

    # Define the dataloader

    dataloader = DataLoader(trainset, batch_size = opt.batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True,drop_last=True)
    # ----------------------------------------
    #                 Training
    # ----------------------------------------

    # Count start time
    prev_time = time.time()
    # For loop training
    for epoch in range(opt.epochs):
        total_loss = 0
        for i, (true_input,stroke_order) in enumerate(dataloader):
            # To device
            true_input = true_input.cuda()
            stroke_order = stroke_order.cuda()
            # print(stroke_order.shape)
            # Train Generator
            optimizer_G.zero_grad()
            pre = generator(true_input,stroke_order)

            # L1 Loss
            # print(true_input.shape)
            # print(pre.shape)
            Pixellevel_L1_Loss = criterion_L1(pre, true_input)

            iters_done = epoch * len(dataloader) + i

            # Overall Loss and optimize
            loss = Pixellevel_L1_Loss
            loss.backward()
            optimizer_G.step()

            # Determine approximate time left
            
            iters_left = opt.epochs * len(dataloader) - iters_done
            time_left = datetime.timedelta(seconds = iters_left * (time.time() - prev_time))
            prev_time = time.time()
            total_loss = Pixellevel_L1_Loss.item() + total_loss

            # # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [Pixellevel L1 Loss: %.4f] Time_left: %s" %
                ((epoch + 1), opt.epochs, i, len(dataloader), Pixellevel_L1_Loss.item(),time_left))
            if iters_done % 100 == 0:
                img_list = [pre, true_input]
                name_list = ['pred', 'input']
                utils.save_sample_png(sample_folder = sample_folder, sample_name = 'Hanzi_sketch_epoch%d' % (epoch + 1), img_list = img_list, name_list = name_list, pixel_max_cnt = 255)

            # Learning rate decrease at certain epochs
            lr = adjust_learning_rate(opt, (epoch + 1), (iters_done + 1), optimizer_G)
            
        save_model(opt, (epoch + 1), (iters_done + 1), len(dataloader), generator)

