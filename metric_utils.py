import numpy as np
import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
def mse(x,y):
    if(x.shape == y.shape):
        return (np.square(x - y)).mean(axis=None)
    else:
        print("DIFFERENT SHAPES")

def psnr(x,y):
    return 20 * np.log10(x.max()) - 10 * np.log10(mse(x,y))

def normalize_matrix(m):
    return (m - m.min()) /(m.max() - m.min())

def center_kernel_np(k):
    roof = k.mean()
    
    columnstart = k.shape[0]
    rowstart = k.shape[1]
    columnend = 0
    rowend = 0
    for i in range(0,k.shape[0]):
        for j in range(0,k.shape[1]):
            if k[i,j] > roof and i < columnstart:
                columnstart = i
            if k[i,j] > roof and j < rowstart:
                
                rowstart = j
            if k[i,j] > roof and i > columnend:
                
                columnend = i
            if k[i,j] > roof and j > rowend:
                
                rowend = j
                

    k_copy = k[columnstart:columnend,rowstart:rowend]
    
    if(k_copy.shape[0] < 4):
        return k
    dim = 0
    if k_copy.shape[0] > k_copy.shape[1]:
        dim = k_copy.shape[0] + 2
    else:
        dim = k_copy.shape[1] + 2
    if dim % 2 == 0:
        dim = dim + 1
    k = np.zeros((dim,dim))
    
    newposy = int((k.shape[0] - k_copy.shape[0]) / 2)
    newposx = int((k.shape[1] - k_copy.shape[1]) / 2)
    k[newposy:newposy + k_copy.shape[0], newposx:newposx + k_copy.shape[1]] = k_copy
    return k


def gen_noisy_images(data_path,noises):
    noisy_path = os.path.join(data_path,"noisy")
    noisy_images = []
    if not os.path.exists(noisy_path):
            os.mkdir(noisy_path)
    
    for i in tqdm(range(0,len(noises))):
        files = list(filter(lambda x: x[-4:] == ".png",os.listdir(data_path)))
        
       
        for f in files:
            img = cv2.imread(os.path.join(data_path,f))
            img = cv2.cvtColor(img,cv2.COLOR_BGRA2GRAY)
            
            row,col= img.shape
            mean = 0
            sigma = noises[i]**0.5
            gauss = np.random.normal(mean,sigma,(row,col))
            
            gauss = gauss.reshape(row,col)
            img_noisy = img + gauss
            noisy_images.append(img_noisy)
            cv2.imwrite(os.path.join(noisy_path,f"{f[:-4]}_{noises[i]}.png"),img_noisy)
    return noisy_images

def gen_noisy_image(img,noises):
    
    noisy_images = []
    for i in tqdm(range(0,len(noises))):
        
        mean = 0
        sigma = noises[i]**0.5
        gauss = np.random.normal(mean,sigma,img.shape)
        
        gauss = gauss.reshape(img.shape)
        img_noisy = img + gauss
        noisy_images.append(img_noisy)
    
    return noisy_images
