import argparse
import os
import numpy as np
import math
import cv2
import itertools

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets import *
from utils import *

import torch.nn as nn
import torch.nn.functional as F
import torch

import scipy.io as scio

os.environ['CUDA_VISIBLE_DEVICES'] ='5'
os.makedirs('images', exist_ok=True)
os.makedirs('saved_models', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--dataset_name', type=str, default="imageNet/train/", help='name of the dataset')
parser.add_argument('--batch_size', type=int, default=8, help='size of the batches')
parser.add_argument('--adv_times', type=float, default=10, help='mult the adv loss of the times')
parser.add_argument('--lr_G', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--lr_D', type=float, default=0.0001, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.9, help='adam: decay of first order momentum of gradient')
parser.add_argument('--decay_epoch', type=int, default=0, help='epoch from which to start lr decay')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_height', type=int, default=256, help='size of image height')
parser.add_argument('--img_width', type=int, default=256, help='size of image width')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=50, help='interval between sampling of images from generators')
parser.add_argument('--checkpoint_interval', type=int, default=-1, help='interval between model checkpoints')
parser.add_argument('--generator_type', type=str, default='unet', help="'resnet' or 'unet'")
parser.add_argument('--cifar_deal_type', type=str, default='en-de', help="'en-en' or 'fc'")
parser.add_argument('--discriminator_type', type=str, default='cdcgan', help="'cgan' or 'cdcgan'")
parser.add_argument('--n_residual_blocks', type=int, default=0, help='number of residual blocks in resnet generator')
parser.add_argument('--hide_skip_pixel', type=bool, default=True, help='skip connection whether to hide the pixels for patches')
parser.add_argument('--region_wise_conv', type=bool, default=True, help='whether use region_wise convolution')

opt = parser.parse_args()

# Loss functions
criterion_GAN = torch.nn.MSELoss()
criterion_translation = torch.nn.L1Loss()
criterion_BCE = torch.nn.BCELoss()

# check whether can use cuda
cuda = True if torch.cuda.is_available() else False

# Calculate output of image discriminator (PatchGAN)
patch_h, patch_w = int(opt.img_height / 2**4), int(opt.img_width / 2**4)
patch = (opt.batch_size, 1, patch_h, patch_w)

# Initialize generator and discriminator
generator = GeneratorResNet(resblocks=opt.n_residual_blocks) if opt.generator_type == 'resnet' else GeneratorUNet(opt=opt, mode=opt.cifar_deal_type)
discriminator = CDiscriminator() if opt.discriminator_type == "cgan" else CDCDiscriminator()
if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion_GAN.cuda()
    criterion_translation.cuda()

# Whether load pretrained models
if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load('saved_models/generator_%d.pth'))
    discriminator.load_state_dict(torch.load('saved_models/discriminator_%d.pth'))
else:
    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

# Loss weight of L1 pixel-wise loss between translated image and real image
lambda_trans = 100

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr_G, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr_D, betas=(opt.b1, opt.b2))

# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Buffers of previously generated samples
Buffer = ReplayBuffer()

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
input_img = Tensor(opt.batch_size, opt.channels, opt.img_height, opt.img_width)
input_cifar = Tensor(opt.batch_size, opt.channels, 32, 32)
input_vec   = Tensor(opt.batch_size, 10)

# Adversarial ground truths
valid = Variable(Tensor(np.ones((opt.batch_size,1))), requires_grad=False)
fake = Variable(Tensor(np.zeros((opt.batch_size,1))), requires_grad=False)

# Dataset loader of ImageNet
transforms_ = [ transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
dataloader = DataLoader(ImageDataset("/media/dsg3/%s" % opt.dataset_name, transforms_=transforms_, mode=["n02102480", "n02105056"]),
                        batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)

# Cifar loader
transforms_cifar = [ 
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
trans_cifar = transforms.Compose(transforms_cifar)                    # use the same transforms as ImageNet

cifar = scio.loadmat("/media/dsg3/datasets/cifar10/cifar_32.mat")     # load the cifar data
cifar_data = cifar['data'].reshape(60000,3,32,32).astype(np.uint8)    # reshape to 
cifar_data = cifar_data.swapaxes(1,2).swapaxes(2,3)                   # swapaxes because the .ToTensor() change [W,H,C] to [C,W,H]

b = []
for i in range(60000):                                                # transforms all cifar picture
    b.append(trans_cifar(cifar_data[i]).unsqueeze(0))    

cifar_data = torch.cat(b,0)

cifar_label = cifar['label']  # size if [60000, 1]
cifar_label = torch.LongTensor(cifar_label)
cifar_label = torch.zeros(60000, 10).scatter_(1, cifar_label, 1)

# Cifar loader
cifarLoader = DataLoader(torch.utils.data.TensorDataset(cifar_data, cifar_label), 
                         batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)

cifarIter = iter(cifarLoader)
cifarN = 0

# Progress logger
logger = Logger(opt.n_epochs, len(dataloader), opt.sample_interval)

# ----------
#  Training
# ----------

for epoch in range(opt.epoch, opt.n_epochs):
    for batch_n, batch in enumerate(dataloader):

        # get the input of cifar
        cifarN += 1
        if cifarN >= len(cifarLoader):
            cifarN = 0
            cifarIter = iter(cifarLoader)

        cifar_img_, cifar_vec_ = cifarIter.next()
        
        cifar_img = Variable(input_cifar.copy_(cifar_img_))
        cifar_vec = Variable(input_vec.copy_(cifar_vec_))
        
        # Set model input
        real_img = Variable(input_img.copy_(batch))
        real_label = Variable(input_vec.copy_(cifar_vec))        
        
        loss_G = []
        loss_D = []
        
        # Generator generates picture and the top_k index
        fake_img, fake_index = generator(real_img, cifar_img, real_label)
        """
        @ fake_img:   list, produced image, length = k(top_k), size = N C W H
        @ fake_index: list, location of embed, length = k(top_k), size = N 1 8 8
        """
        
        # for every top_k index to calculate loss 
        for index,image_index in zip(fake_index,fake_img):
            """
            @ index:       top_k
            @ image_index: fake_images produced
            """
            loss_l1 = 0.            
            index_value        = index.cpu().data.numpy()      # transform the location to numpy
            fake_index_list    = np.argwhere(index_value == 1) # to find which pixel insert cifar image
            fake_img_dis = []                                  # to store the cifar patches produced
            fake_img_np = image_index.cpu().data.numpy()       # fake_image produced to numpy
            center =  []                                       # to find the location center 

            # for every picture to select its center
            for i,f_index in zip(range(0,opt.batch_size),fake_index_list): 
                center.append((f_index[3] * 32 + 16, f_index[2] * 32 + 16))
                fake_img_dis.append(image_index[i:i+1, :, f_index[2]*32:f_index[2]*32+32, f_index[3]*32:f_index[3]*32+32])
                
                # mask, similar to face mask, to stand for which patch is cifar iamge produced
                mask = np.ones((opt.img_height,opt.img_width))
                mask[int(f_index[2]*32):int(f_index[2]*32+32), int(f_index[3]*32):int(f_index[3]*32+32)] = 0
                mask = Tensor(mask)
            
            # To make ground truth, the real cifar image  
            cifar_embed = (cifar_img*0.5 + 0.5) * 255
            cifar_embed = cifar_embed.cpu().data.numpy()
            cifar_embed = cifar_embed.swapaxes(1,2).swapaxes(2,3)                

            # To make ground truth, the real ImageNet image 
            img_label = (real_img*0.5 + 0.5) * 255
            img_label = img_label.cpu().data.numpy()
            img_label = img_label.swapaxes(1,2).swapaxes(2,3)
 
            # make use of Possion, to make ground truth
            label  = Tensor(possion(cifar_embed, img_label, center)).cuda()
            label  = label / 255.
            label  = (label - 0.5) / 0.5

            # ---------------------
            #  Train Generator
            # ---------------------
            optimizer_G.zero_grad()
          
            # calculate loss of l1
            loss_l1 = criterion_translation(image_index, label) 
                   
            # dis_input contains patches of cifar produced 
            dis_input = torch.cat(fake_img_dis,0).cuda()

            # calculate g_adv 
            pred_fake = discriminator(dis_input, cifar_vec)
            loss_g_adv = criterion_BCE(pred_fake,valid) * opt.adv_times
         
            # G Total loss
            loss_g = loss_l1 + loss_g_adv
            loss_g.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # calculate loss of dis 
            pred_fake = discriminator(dis_input, cifar_vec)
            loss_d_fake = criterion_BCE(pred_fake.detach(),fake) * opt.adv_times
         
            pred_real   = discriminator(cifar_img, cifar_vec)
            loss_d_real = criterion_BCE(pred_real, valid) * opt.adv_times
            
            # D Total loss
            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            optimizer_D.step()
        '''# backward the loss
        for k in range(0,len(loss_D)):
            optimizer_G.zero_grad()
            if k == 0:
                loss_G[k].backward(retain_graph=True)
            else:
                loss_G[k].backward()
            optimizer_G.step()
            
            optimizer_D.zero_grad()
            if k == 0:
                loss_D[k].backward(retain_graph=True)
            else:
                loss_D[k].backward()
            optimizer_D.step()
        ''' 
        # --------------
        #  Log Progress
        # --------------

        logger.log({'loss_G':  loss_g, 
                    'loss_l1': loss_l1,
                    'loss_d_fake': loss_d_fake,
                    'loss_d_real': loss_d_real, 
                    'loss_g_adv' : loss_g_adv},
                    images={
                    'real': real_img,
                    'fake': fake_img[0],
                    'cifar':cifar_img},
                    epoch=epoch, batch=batch_n)

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D.step()

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), 'saved_models/generator_%d.pth' % epoch)
        torch.save(discriminator.state_dict(), 'saved_models/discriminator_%d.pth' % epoch)
