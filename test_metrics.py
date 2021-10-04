import argparse
import os
import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np

from metric_utils import mse,psnr
from image_similarity_measures.quality_metrics import ssim,fsim,issm,sre,uiq,sam
from pytorch_msssim import ms_ssim,MS_SSIM
import lpips
parser = argparse.ArgumentParser()
parser.add_argument('--im0', type=str)
parser.add_argument('--im1', type=str)
parser.add_argument('--im2', type=str)
opt = parser.parse_args()

im0 = cv2.imread(opt.im0)[:,:]

im1 = cv2.imread(opt.im1)[:,:]
if opt.im2 != None: 
    im2 = cv2.imread(opt.im2)[:,:]

im0g = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
im1g = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
if opt.im2 != None: 
    im2g = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

im0 = (im0 - im0.min()) /(im0.max() - im0.min())
im1 = (im1 - im1.min()) /(im1.max() - im1.min())
if opt.im2 != None: 
    im2 = (im2 - im2.min()) /(im2.max() - im2.min())


im0t = np.expand_dims(np.moveaxis(im0,-1,0),0)
im0t = torch.from_numpy(im0t).float()

im1t = np.expand_dims(np.moveaxis(im1,-1,0),0)
im1t = torch.from_numpy(im1t).float()
print(im0.shape)
print(f"SSIM: {ssim(im0g,im1g)}")
print(f"MSE: {mse(im0g,im1g) /(im0g.shape[0]*im0g.shape[1])}")
print(f"PSNR: {psnr(im0g,im1g)}")
print(f"FSIM: {fsim(im0,im1)}")
#print(f"SRE: {sre(im0,im1)}")
#print(f"UIQ: {uiq(im0,im1)}")
#print(f"SAM: {sam(im0,im1)}")
#print(f"MS-SSIM: {ms_ssim(im0t,im1t)}")


fig = plt.figure()

ax = fig.add_subplot(1,3,1)
ax.set_title("Immagine originale")
plt.imshow(im0)
print("b")
ax = fig.add_subplot(1,3,2)
ax.set_title("Selfdeblur")
plt.imshow(im1)
ax.text(10,310,f"ssim:{ssim(im0g,im1g)}\nmse: {mse(im0g,im1g)/(255**2)}\npsnr: {psnr(im0g,im1g)}")
if opt.im2 != None: 
    ax = fig.add_subplot(1,3,3)
    ax.set_title("Selfdeblur con kernel ridimensionato")
    ax.text(10,310,f"ssim:{ssim(im0g,im2g)}\nmse: {mse(im0g,im2g)/(255**2)}\npsnr: {psnr(im0g,im2g)}")
    plt.imshow(im2)

plt.show()