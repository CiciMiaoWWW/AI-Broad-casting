import argparse
import os
import trainers.trainer as trainer
import trainers.trainer_pair as trainer_pair
import trainers.trainer_LXMERT as trainer_LXMERT

if __name__ == "__main__":
    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # Pre-train, saving, and loading parameters
    parser.add_argument('--mode', type = str, default = 'Single', help = 'Single / LXMERT /  Pair') # for second stage, change it to False
    parser.add_argument('--save_mode', type = str, default = 'epoch', help = 'saving mode, and by_epoch saving is recommended')
    parser.add_argument('--save_by_epoch', type = int, default = 25, help = 'interval between model checkpoints (by epochs)')
    parser.add_argument('--save_best_model', type = bool, default = True, help = 'save best model ot not')
    parser.add_argument('--save_by_iter', type = int, default = 10000000, help = 'interval between model checkpoints (by iterations)')
    parser.add_argument('--save_name_mode', type = bool, default = True, help = 'True for concise name, and False for exhaustive name')
    parser.add_argument('--load_name', type = str, default = 'Pre_model', help = 'load the pre-trained model with certain epoch')
    # GPU parameters
    parser.add_argument('--multi_gpu', type = bool, default = False, help = 'True for more than 1 GPU')
    parser.add_argument('--gpu_ids', type = str, default = '2,3', help = 'gpu_ids: e.g. 0  0,1  0,1,2  use -1 for CPU')
    parser.add_argument('--cudnn_benchmark', type = bool, default = True, help = 'True for unchanged input data type')
    # Training parameters
    parser.add_argument('--use_checkpoint', type = bool, default = False, help = 'True for resuming')
    parser.add_argument('--checkpoint_load_path', type = str, default = 'checkpoint_path', help = 'checkpoint path')
    parser.add_argument('--epochs', type = int, default = 500, help = 'number of epochs of training') # for second stage, change it to 30
    parser.add_argument('--batch_size', type = int, default = 4, help = 'size of the batches')
    parser.add_argument('--crop_size', type = int, default = 256, help = 'crop size')
    parser.add_argument('--lr_g', type = float, default = 0.0001, help = 'Adam: learning rate for G')
    parser.add_argument('--lr_d', type = float, default = 0.0004, help = 'Adam: learning rate for D')
    parser.add_argument('--b1', type = float, default = 0.5, help = 'Adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type = float, default = 0.999, help = 'Adam: decay of second order momentum of gradient')
    parser.add_argument('--b3', type = float, default = 0.9, help = 'Adam: decay of second order momentum of gradient')
    parser.add_argument('--weight_decay', type = float, default = 0, help = 'weight decay for optimizer')
    parser.add_argument('--lr_decrease_mode', type = str, default = 'epoch', help = 'lr decrease mode, by_epoch or by_iter')
    parser.add_argument('--lr_decrease_epoch', type = int, default = 15, help = 'lr decrease at certain epoch and its multiple')
    parser.add_argument('--lr_decrease_iter', type = int, default = 1600, help = 'lr decrease at certain epoch and its multiple')
    parser.add_argument('--lr_decrease_factor', type = float, default = 0.9, help = 'lr decrease factor')
    parser.add_argument('--num_workers', type = int, default = 8, help = 'number of cpu threads to use during batch generation')
    parser.add_argument('--lambda_tv', type = float, default = 0.5, help = 'coefficient for TV Loss')
    parser.add_argument('--lambda_percep', type = float, default = 1, help = 'coefficient for Perceptual Loss')
    parser.add_argument('--lambda_gan', type = float, default = 0.01, help = 'coefficient for GAN Loss')
    # Initialization parameters
    parser.add_argument('--pad', type = str, default = 'reflect', help = 'pad type of networks')
    parser.add_argument('--norm', type = str, default = 'none', help = 'normalization type of networks')
    parser.add_argument('--activ', type = str, default = 'lrelu', help = 'acti')
    parser.add_argument('--hidden_act', type = str, default = "gelu", help = 'acti')
    parser.add_argument('--activ_g', type = str, default = 'relu', help = 'activation type of generator')
    parser.add_argument('--activ_d', type = str, default = 'lrelu', help = 'activation type of discriminator')
    parser.add_argument('--in_channels', type = int, default = 3, help = '1 for colorization, 3 for other tasks')
    parser.add_argument('--out_channels', type = int, default = 3, help = '2 for colorization, 3 for other tasks')
    parser.add_argument('--max_position_embeddings', type = int, default = 29, help = '2 for colorization, 3 for other tasks')
    parser.add_argument('--start_channels', type = int, default = 16, help = 'start channels for the main stream of generator')
    parser.add_argument('--num_attention_heads', type = int, default = 8, help = 'start channels for the main stream of generator')
    parser.add_argument('--intermediate_size', type = int, default = 2048, help = 'start channels for the main stream of generator')
    parser.add_argument('--hidden_size', type = int, default = 512, help = 'hiden size for transformer')
    parser.add_argument('--init_type', type = str, default = 'xavier', help = 'initialization type of networks')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'initialization gain of networks')
    parser.add_argument('--hidden_dropout_prob', type = float, default = 0.5, help = 'drop pro')
    parser.add_argument('--attention_probs_dropout_prob', type = float, default = 0.1, help = 'drop pro')
    # GAN parameters
    parser.add_argument('--gan_mode', type = str, default = 'LSGAN', help = 'type of GAN: [LSGAN | WGAN], WGAN is recommended')
    parser.add_argument('--additional_training_d', type = int, default = 1, help = 'number of training D more times than G')
    # Dataset parameters
    parser.add_argument('--task', type = str, default = 'try', help = 'the specific task of the system')
    parser.add_argument('--angle_aug', type = bool, default = True, help = 'data augmentation')
    parser.add_argument('--in_root', type = str, default = './txt_folder/train.txt', help = 'input image path list')
    parser.add_argument('--val_root', type = str, default = './txt_folder/test.txt', help = 'val image path list')
    parser.add_argument('--test_root', type = str, default = './txt_folder/test.txt', help = 'test image path list')
    parser.add_argument('--sample_path', type = str, default = './sample', help = 'val image save root')
    parser.add_argument('--checkpoint_path', type = str, default = './checkpoints/', help = 'checkpoint save baseroot')
    opt = parser.parse_args()

    # ----------------------------------------
    #        Choose CUDA visible devices
    # ----------------------------------------
    if opt.multi_gpu == True:
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
        print('Multi-GPU mode, %s GPUs are used' % (opt.gpu_ids))
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('Single-GPU mode')
    
    # ----------------------------------------
    #       Choose pre / continue train
    # ----------------------------------------
    if opt.mode=='Single':
        print('My-training settings: [Epochs: %d] [Batch size: %d] [Learning rate: %.4f] [Mode: %s]'
            % (opt.epochs, opt.batch_size, opt.lr_g, opt.mode))
        trainer.MyDNN(opt)
    elif opt.mode=='LXMERT':
        print('My-training settings: [Epochs: %d] [Batch size: %d] [Learning rate: %.4f] [Mode: %s]'
            % (opt.epochs, opt.batch_size, opt.lr_g, opt.mode))
        trainer_LXMERT.MyDNN(opt)
    else:
        print('My-training settings: [Epochs: %d] [Batch size: %d] [Learning rate: %.4f] [Mode: %s]'
            % (opt.epochs, opt.batch_size, opt.lr_g, opt.mode))
        trainer_pair.MyDNN(opt)  
