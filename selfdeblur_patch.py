from __future__ import print_function
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
from networks.skip import skip
from networks.fcn import fcn
import cv2
import torch
import torch.optim
import glob
from skimage.io import imread
from skimage.io import imsave
import warnings
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR
from utils.common_utils import *
from SSIM import SSIM
from metric_utils import gen_noisy_image
import time
from metric_utils import center_kernel_np,normalize_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--num_iter', type=int, default=3000, help='number of epochs of training')
parser.add_argument('--img_size', type=int, default=[256, 256], help='size of each image dimension')
parser.add_argument('--kernel_size', type=int, default=[25, 25], help='size of blur kernel [height, width]')
parser.add_argument('--data_path', type=str, default="datasets/levin/", help='path to blurry image')
parser.add_argument('--save_path', type=str, default="results/levin/", help='path to save results')
parser.add_argument('--save_frequency', type=int, default=1000, help='lfrequency to save results')
opt = parser.parse_args()
#print(opt)

torch.backends.cudnn.enabled = True
dtype = torch.cuda.FloatTensor

warnings.filterwarnings("ignore")

files_source = glob.glob(os.path.join(opt.data_path, '*.png'))
files_source.sort()
save_path = opt.save_path
os.makedirs(save_path, exist_ok=True)
def total_variation_loss(img, weight):
     bs_img, c_img, h_img, w_img = img.size()
     tv_h = torch.pow(img[:,:,1:,:]-img[:,:,:-1,:], 2).sum()
     tv_w = torch.pow(img[:,:,:,1:]-img[:,:,:,:-1], 2).sum()
     return weight*(tv_h+tv_w)/(bs_img*c_img*h_img*w_img)
def tv_loss2(img,weight):
    bs_img, c_img, h_img, w_img = img.size()
    tv_h = torch.pow(img[:,:,1:,:]-img[:,:,:-1,:], 1).sum()
    tv_w = torch.pow(img[:,:,:,1:]-img[:,:,:,:-1], 1).sum()
    return weight*(tv_h+tv_w)


def get_patches(x,overlap):
    if x.max() > 1:
        x = (x - x.min()) / (x.max() - x.min())
    w = int(x.shape[2]/2)
    h = int(x.shape[1]/2)
    p0 = x[:,0:h,0:w+overlap]
    p1 = x[:,h-overlap:,0:w]
    p2 = x[:,0:h+overlap,w:]
    p3 = x[:,h:,w-overlap:]
    return [p0,p1,p2,p3]

def resolve_patches(p,overlap,h,w):
    p0 = p[0]
    p1 = p[1]
    p2 = p[2]
    p3 = p[3]
    y = np.zeros((256,256,3))
    ov0 = (p0[-overlap:,:-overlap,:] + p1[:overlap,:,:])/2
    ov1 = (p0[:,-overlap:,:] + p2[:-overlap,:overlap,:])/2
    ov2 = (p2[-overlap:,:,:] + p3[:overlap:,overlap:,:])/2
    ov3 = (p1[overlap:,-overlap:,:] + p3[:,:overlap,:])/2

    y[0:h-overlap,0:w,:] = p0[0:h-overlap,0:w,:] 
    y[0:h,w+overlap:,:] = p2[0:h,overlap:,:]

    y[h:,:w-overlap,:] = p1[overlap:,:w-overlap,:]
    y[h+overlap:,w:,:] = p3[overlap:,overlap:,:]

    y[h-overlap:h,0:w,:] = ov0
    y[0:h,w:w+overlap,:] = ov1
    y[h:h+overlap,w:,:] = ov2
    y[h:,w-overlap:w,:] = ov3
    return y


def selfdeblur(p,ii,imgname):
    INPUT = 'noise'
    pad = 'reflection'
    LR = 0.01
    num_iter = opt.num_iter
    reg_noise_std = 0.001
    y = np_to_torch(p).type(dtype)

    img_size = p.shape
    print(imgname)
    # ######################################################################
    padh, padw = opt.kernel_size[0]-1, opt.kernel_size[1]-1
    opt.img_size[0], opt.img_size[1] = img_size[1]+padh, img_size[2]+padw
    '''
    x_net:
    '''
    input_depth = 8
    net_input = get_noise(input_depth, INPUT, (opt.img_size[0], opt.img_size[1])).type(dtype)
    net = skip( input_depth, 1,
                num_channels_down = [128, 128, 128, 128, 128],
                num_channels_up   = [128, 128, 128, 128, 128],
                num_channels_skip = [16, 16, 16, 16, 16],
                upsample_mode='bilinear',
                need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')
    net = net.type(dtype)
    '''
    k_net:
    '''
    n_k = 200
    net_input_kernel = get_noise(n_k, INPUT, (1, 1)).type(dtype)
    net_input_kernel.squeeze_()
    net_kernel = fcn(n_k, opt.kernel_size[0]*opt.kernel_size[1])
    net_kernel = net_kernel.type(dtype)
    # Losses
    mse = torch.nn.MSELoss().type(dtype)
    ssim = SSIM().type(dtype)
    # optimizer
    optimizer = torch.optim.Adam([{'params':net.parameters()},{'params':net_kernel.parameters(),'lr':1e-4}], lr=LR)
    scheduler = MultiStepLR(optimizer, milestones=[2000, 3000, 4000], gamma=0.5)  # learning rates
    
    ### start SelfDeblur
    for step in tqdm(range(num_iter)):
        # input regularization
        #net_input = net_input_saved + reg_noise_std*torch.zeros(net_input_saved.shape).type_as(net_input_saved.data).normal_()
        
        
        # change the learning rate
        scheduler.step(step)
        optimizer.zero_grad()
        # get the network output
        out_x = net(net_input)
        out_k = net_kernel(net_input_kernel)
    
        out_k_m = out_k.view(-1,1,opt.kernel_size[0],opt.kernel_size[1])
        # print(out_k_m)
        out_y = nn.functional.conv2d(out_x, out_k_m, padding=0, bias=None)
        if step < 1000:
            total_loss = mse(out_y,y) 
        else:
            total_loss = 1-ssim(out_y, y) 
        total_loss.backward()
        optimizer.step()
        if (step+1) % opt.save_frequency == 0:
            #print('Iteration %05d' %(step+1))
            save_path = os.path.join(opt.save_path, '%s_%d_x.png'%(imgname,ii))
            out_x_np = torch_to_np(out_x)
            out_x_np = out_x_np.squeeze()
            out_x_np = out_x_np[padh//2:padh//2+img_size[1], padw//2:padw//2+img_size[2]]
            imsave(save_path, out_x_np)
            save_path = os.path.join(opt.save_path, '%s_%d_k.png'%(imgname,ii))
            out_k_np = torch_to_np(out_k_m)
            out_k_np = out_k_np.squeeze()
            out_k_np /= np.max(out_k_np)
            imsave(save_path, out_k_np)
            torch.save(net, os.path.join(opt.save_path, '%s_%d_xnet.pth'%(imgname,ii)))
            torch.save(net_kernel, os.path.join(opt.save_path, '%s_%d_knet.pth'%(imgname,ii)))

def nonblind_selfdeblur(p,ii,imgname):
    INPUT = 'noise'
    pad = 'reflection'
    LR = 0.01
    num_iter = opt.num_iter
    reg_noise_std = 0.001
    padh, padw = opt.kernel_size[0]-1, opt.kernel_size[1]-1
    img_size = p.shape
    opt.img_size[0], opt.img_size[1] = img_size[1]+padh, img_size[2]+padw
    y = np_to_torch(p).type(dtype)
    '''
    x_net:
    '''
    input_depth = 8

    net_input = get_noise(input_depth, INPUT, (opt.img_size[0], opt.img_size[1])).type(dtype)

    net = skip( input_depth, 1,
                num_channels_down = [128, 128, 128, 128, 128],
                num_channels_up   = [128, 128, 128, 128, 128],
                num_channels_skip = [16, 16, 16, 16, 16],
                upsample_mode='bilinear',
                need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

    net = net.type(dtype) 

    # Losses
    mse = torch.nn.MSELoss().type(dtype)
    ssim = SSIM().type(dtype)

    # optimizer
    optimizer = torch.optim.Adam([{'params':net.parameters()}], lr=LR)
    scheduler = MultiStepLR(optimizer, milestones=[700, 800, 900], gamma=0.5)  # learning rates

    # initilization inputs
    net_input_saved = net_input.detach().clone()

    ### start SelfDeblur
    for step in tqdm(range(num_iter)):

        # input regularization
        net_input = net_input_saved + reg_noise_std*torch.zeros(net_input_saved.shape).type_as(net_input_saved.data).normal_()

        # change the learning rate
        scheduler.step(step)
        optimizer.zero_grad()

        # get the network output
        out_x = net(net_input)

        # print(out_k_m)
        out_y = nn.functional.conv2d(out_x, out_k, padding=0, bias=None)

        total_loss = 1 - ssim(out_y, y)  
        total_loss.backward()
        optimizer.step()

        if (step+1) % opt.save_frequency == 0:
            #print('Iteration %05d' %(step+1))

            save_path = os.path.join(opt.save_path, '%s_%d_x.png'%(imgname,ii))
            out_x_np = torch_to_np(out_x)
            out_x_np = out_x_np.squeeze()
            out_x_np = out_x_np[padh//2:padh//2+img_size[1], padw//2:padw//2+img_size[2]]
            imsave(save_path, out_x_np)

            torch.save(net, os.path.join(opt.save_path, "%s_%d_xnet.pth" % (imgname,ii)))

# start#image
for f in files_source:
   

    path_to_image = f
    imgname = os.path.basename(f)
    imgname = os.path.splitext(imgname)[0]


    _, imgs = get_image(path_to_image, -1, odd=False) # load image and convert to np.
    
    
    #imgs = gen_noisy_image(imgs,[0.01])[0]
    patches = get_patches(imgs,25)
    
    selfdeblur(patches[0],0,imgname)
    #Load kernel
    out_k = cv2.imread(os.path.join(opt.save_path, '%s_%d_k.png'%(imgname,0)),cv2.IMREAD_GRAYSCALE)
    out_k = center_kernel_np(out_k)
    out_k = np.expand_dims(np.float32(out_k/255.),0)
    out_k = np_to_torch(out_k).type(dtype)
    out_k = torch.clamp(out_k, 0., 1.)
    out_k /= torch.sum(out_k)
    opt.kernel_size = [out_k.shape[2], out_k.shape[3]]
    ii = 1
    print(len(patches[1:]))
    for p in patches[1:]:
        nonblind_selfdeblur(p,ii,imgname)
        
        ii+=1
    r_patches = []
    for i in range(4):
        r_patches.append(normalize_matrix(cv2.imread(os.path.join(opt.save_path, '%s_%d_x.png'%(imgname,i)))))
    res = resolve_patches(r_patches,25,128,128)
    cv2.imwrite(os.path.join(opt.save_path,"%s_patched.png" % imgname,res*255))