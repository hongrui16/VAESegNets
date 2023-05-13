import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
# from utils import *
# from modules import UNet
import logging
import torch
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader
import os, sys
from torch.optim import Adam, SGD
from torch.utils.tensorboard import SummaryWriter
import glob
import numpy



class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)




class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        x = self.maxpool_conv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            # DoubleConv(int(in_channels*1.5), in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

    def forward(self, x, skip_x):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        # print('x.shape', x.shape)
        x = self.conv(x)
        return x 
    
class onlyUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            # DoubleConv(int(in_channels*1.5), in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x 

class catUp(nn.Module):
    def __init__(self, in_channels, out_channels, residual=True):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels*2, in_channels*2, residual=residual),
            DoubleConv(in_channels*2, out_channels, in_channels // 2),
        )

    def forward(self, x, skip_x):        
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        x = self.up(x)
        return x 
    
class LikeUNet(nn.Module):
    def __init__(self, c_in=3, c_out=2):
        super().__init__()
        self.inc = DoubleConv(c_in, c_in)
        self.down1 = Down(c_in, 16) #128
        self.down2 = Down(16, 32) #64
        self.down3 = Down(32, 64) #32
        self.down4 = Down(64, 128) #16
        self.down5 = Down(128, 256) #8*256
        self.down6 = Down(256, 512) #4

        self.bot1 = DoubleConv(512, 512)
        # self.bot3 = DoubleConv(512, 256)
        self.up0 = catUp(512, 256) #8
        self.up1 = catUp(256, 128) #16
        self.up2 = catUp(128, 64) #32
        self.up3 = catUp(64, 32) #64
        self.up4 = catUp(32, 16) #128
        self.catup = catUp(16, c_out)


    def forward(self, x):
        # print(f'forward    x: {x.shape}, t: {t.shape}')
        x = self.inc(x)
        xd1 = self.down1(x) #16*128
        xd2 = self.down2(xd1) #32*64
        xd3 = self.down3(xd2) #64*32
        xd4 = self.down4(xd3) #128*16
        xd5 = self.down5(xd4) #256*8
        xd6 = self.down6(xd5) #512*4
        # print(f'forward    x1: {x1.shape}, t: {t.shape}')
        x = self.bot1(xd6)
        # print(f'forward    x: {x.shape}, xd6: {xd6.shape}') #x: torch.Size([32, 512, 4, 4]), xd6: torch.Size([32, 512, 4, 4])
        # print(f'forward    x: {x.shape}, xd5: {xd5.shape}') #x: torch.Size([32, 512, 4, 4]), xd5: torch.Size([32, 256, 8, 8])

        xu0 = self.up0(x, xd6) #8
        xu1 = self.up1(xu0, xd5)
        xu2 = self.up2(xu1, xd4)
        xu3 = self.up3(xu2, xd3)
        xu4 = self.up4(xu3, xd2)
        xu4 = self.catup(xu4, xd1)
        return xu4


    
class VAEUNet(nn.Module):
    def __init__(self, c_in=3, c_out=2):
        super().__init__()
        self.inc = DoubleConv(c_in, c_in)
        self.down1 = Down(c_in, 16) #128
        self.down2 = Down(16, 32) #64
        self.down3 = Down(32, 64) #32
        self.down4 = Down(64, 128) #16
        self.down5 = Down(128, 256) #8*256
        self.down6 = Down(256, 512) #4

        self.bot1 = DoubleConv(512, 512)

    
        self.up0 = catUp(512, 256) #8
        self.up1 = catUp(256, 128) #16
        self.up2 = catUp(128, 64) #32
        self.up3 = catUp(64, 32) #64
        self.up4 = catUp(32, 16) #128
        self.catup = catUp(16, c_out)

        self.mu = Down(512, 64) #2
        self.logvar = Down(512, 64) #2
        self.bot2 = DoubleConv(64, 512)

        self.onlyup1 = onlyUp(512, 256) #4
        self.onlyup2 = onlyUp(256, 128) #8
        self.onlyup3 = onlyUp(128, 64) #16
        self.onlyup4 = onlyUp(64, 32) #32
        self.onlyup5 = onlyUp(32, 16) #64
        self.onlyup6 = onlyUp(16, 8) #128
        self.onlyup7 = onlyUp(8, 3) #256
        

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def decoder(self, z):
        z = z.view(z.size(0), 64, 2, 2)
        p = self.bot2(z)
        p = self.onlyup1(p)
        p = self.onlyup2(p)
        p = self.onlyup3(p)
        p = self.onlyup4(p)
        p = self.onlyup5(p)
        p = self.onlyup6(p)
        p = self.onlyup7(p)
        return p

    def forward(self, x):
        # print(f'forward    x: {x.shape}, t: {t.shape}')
        x = self.inc(x)
        xd1 = self.down1(x) #16*128
        xd2 = self.down2(xd1) #32*64
        xd3 = self.down3(xd2) #64*32
        xd4 = self.down4(xd3) #128*16
        xd5 = self.down5(xd4) #256*8
        xd6 = self.down6(xd5) #512*4
        # print(f'forward    x1: {x1.shape}, t: {t.shape}')
        bot1 = self.bot1(xd6)
        # print(f'forward    x: {x.shape}, xd6: {xd6.shape}') #x: torch.Size([32, 512, 4, 4]), xd6: torch.Size([32, 512, 4, 4])
        # print(f'forward    x: {x.shape}, xd5: {xd5.shape}') #x: torch.Size([32, 512, 4, 4]), xd5: torch.Size([32, 256, 8, 8])

        xu0 = self.up0(bot1, xd6) #8
        xu1 = self.up1(xu0, xd5)
        xu2 = self.up2(xu1, xd4)
        xu3 = self.up3(xu2, xd3)
        xu4 = self.up4(xu3, xd2)
        seg = self.catup(xu4, xd1)


        mu = self.mu(bot1)
        mu = mu.view(mu.size(0), -1)
        logvar = self.logvar(bot1)
        logvar = logvar.view(logvar.size(0), -1)
        z = self.reparameterize(mu, logvar)

        return seg, self.decoder(z), mu, logvar



class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 2, padding = 1):
        super().__init__()
        self.basic_conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride = stride,  padding=padding, bias=False),
        nn.BatchNorm2d(out_channels), 
        nn.LeakyReLU(0.2, inplace=True),)

    def forward(self, x):
        return self.basic_conv(x)





class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, skip_connect = True):
        super().__init__()
        if skip_connect:
            in_channels *= 2
        self.basic_convTrans = nn.Sequential(
        # nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 3, stride=2, padding=1), ### 2*2*256 -> 4*4*128
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 2, stride=2), ### 2*2*256 -> 4*4*128
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=True),)

        self.conv = nn.Sequential(
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride = 1, padding=1, bias=False),
        nn.BatchNorm2d(out_channels), 
        nn.LeakyReLU(0.2, inplace=True),)


    def forward(self, x, skip_x = None):
        if skip_x is None:
            x = self.basic_convTrans(x)
            return x
        x = torch.cat([skip_x, x], dim=1)
        x = self.basic_convTrans(x)
        x = self.conv(x)
        return x 




class OutputLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
        nn.Conv2d(in_channels*2, in_channels, kernel_size=3, stride = 1, padding=1, bias=False),
        nn.BatchNorm2d(in_channels), 
        nn.LeakyReLU(0.2, inplace=True),)

        self.convTrans = nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 2, stride=2), ### 2*2*256 -> 4*4*128
        nn.Tanh(),)

        


    def forward(self, x, skip_x):
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        x = self.convTrans(x)
        
        return x 



class bscGeneator(nn.Module):
    def __init__(self, c_in=6, c_out=1):
        super().__init__()

        self.down1 = DownSample(c_in, 64) #128
        self.down2 = DownSample(64, 64) #64
        self.down3 = DownSample(64, 64) #32
        self.down4 = DownSample(64, 64) #16
        self.down5 = DownSample(64, 64) #8
        self.down6 = DownSample(64, 128) #4
        self.down7 = DownSample(128, 256) #2
        self.down8 = DownSample(256, 512) #1



        self.up1 = UpSample(512, 256, False) #2
        self.up2 = UpSample(256, 128) #4
        self.up3 = UpSample(128, 64) #8
        self.up4 = UpSample(64, 64) #16
        self.up5 = UpSample(64, 64) #32
        self.up6 = UpSample(64, 64) #64
        self.up7 = UpSample(64, 64) #128

        self.outc =  OutputLayer(64, c_out) #128


    def forward(self, x):

        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.down6(x5)
        x7 = self.down7(x6)
        x8 = self.down8(x7)


        x = self.up1(x8)
        # print('x.shape', x.shape, 'x8.shape', x8.shape) #x.shape torch.Size([16, 256, 2, 2]) x8.shape torch.Size([16, 512, 1, 1])
        # print('x.shape', x.shape, 'x7.shape', x7.shape) 
        x = self.up2(x, x7)
        x = self.up3(x, x6)
        x = self.up4(x, x5)
        x = self.up5(x, x4)
        x = self.up6(x, x3)
        x = self.up7(x, x2)
        output = self.outc(x, x1)
        return output





class bscDiscriminator(nn.Module):
    def __init__(self, c_in=7, c_out=1, img_size = 256):
        super().__init__()

        self.down1 = DownSample(c_in, 16) #128
        self.down2 = DownSample(16, 32) #64
        self.down3 = DownSample(32, 64) #32
        self.down4 = DownSample(64, 128) #16
        img_shape = (128, 16, 16)

        self.fc = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 128),
            nn.Linear(128, c_out),
            nn.Sigmoid()
        )

    def forward(self, img):
        # print('bscDiscriminator', img.shape)
        out = self.down1(img)
        out = self.down2(out)
        out = self.down3(out)
        out = self.down4(out)
        out = self.fc(out.view(out.size(0), -1))
        return out



# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()


#         self.conv_blocks = nn.Sequential(
#             nn.Conv2d(6, 64, 3, stride=2, padding=1), ### 256*256*6-> 128*128*64
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2, inplace=True),
            
#             nn.Conv2d(64, 64, 3, stride=2, padding=1), ### 128*128*64 -> 64*64*64
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2, inplace=True),

#             nn.Conv2d(64, 64, 3, stride=2, padding=1), ### 64*64*64 -> 32*32*64
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2, inplace=True),

#             nn.Conv2d(64, 64, 3, stride=2, padding=1), ### 32*32*64 -> 16*16*64
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2, inplace=True),

#             nn.Conv2d(64, 128, 3, stride=2, padding=1), ### 16*16*64 -> 8*8*128
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2, inplace=True),

#             nn.Conv2d(128, 128, 3, stride=2, padding=1), ### 8*8*128 -> 4*4*128
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2, inplace=True),

#             nn.Conv2d(128, 256, 3, stride=2, padding=1), ### 4*4*128 -> 2*2*256
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2, inplace=True),
            
#             nn.Conv2d(256, 512, 3, stride=2, padding=1), ### 2*2*256 -> 1*1*512
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2, inplace=True),


#             nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1), ### 1*1*512 -> 2*2*256
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2, inplace=True),

#             nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1), ### 2*2*256 -> 4*4*128
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2, inplace=True),

#             nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1), ### 4*4*128 -> 8*8*64
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2, inplace=True),

#             nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1), ### 8*8*64 ->16*16*64
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2, inplace=True),

#             nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1), ### 16*16*64 -> 16*16*64
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2, inplace=True),

#             nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1), ### 32*32*64 -> 64*64*64
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2, inplace=True),

#             nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1), ### 64*64*64 -> 128*128*64
#             nn.Tanh(),
#         )

#     def forward(self, x):
#         img = self.conv_blocks(x)
#         return img





























