import torch.nn as nn
from .common import *


def cnn(flag, ksize):
    kernel = 5
    pad = (kernel-1)//2
    model = nn.Sequential()
    model.add(nn.Conv2d(in_channels=1, out_channels=8,
                        kernel_size=3, padding=1, bias=False))                       
    model.add(nn.ReLU())
    model.add(nn.Conv2d(in_channels=8, out_channels=16,
                        kernel_size=kernel, padding=pad, bias=False))
    model.add(nn.ReLU())
    model.add(nn.Conv2d(in_channels=16, out_channels=32,
                        kernel_size=kernel, padding=pad, bias=False))
    model.add(nn.ReLU())
    model.add(nn.Conv2d(in_channels=32, out_channels=16,
                        kernel_size=kernel, padding=pad, bias=False))
    model.add(nn.ReLU())
    model.add(nn.Conv2d(in_channels=16, out_channels=8,
                        kernel_size=kernel, padding=pad, bias=False))
    model.add(nn.ReLU())
    model.add(nn.Conv2d(in_channels=8, out_channels=1,
                        kernel_size=kernel, padding=pad, bias=False))
    model.add(nn.ReLU())
    model.add(nn.Flatten())
    if (flag):
        model.add(nn.Linear(ksize, 1000))
        model.add(nn.ReLU())
        model.add(nn.Linear(1000, ksize))
        model.add(nn.Softmax())
    return model