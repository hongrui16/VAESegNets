import numpy as np
from sklearn.metrics import confusion_matrix  
import sys
import time
import os
import torch
import random
import shutil
import cv2
import scipy
import torch
from PIL import Image
import math
from math import pi
import imageio
import pytz
import datetime
import csv
from matplotlib import pyplot as plt
import pandas as pd
from utils import *

def load_weight_plot_loss():
    g_weight_filepath = 'logs/VAEUNet/run_0/VAEUNet_checkpoint.pth.tar'
    checkpoint = torch.load(g_weight_filepath, map_location=torch.device('cpu'))

    g_trainingEpoch_loss = checkpoint['trainingEpoch_loss']
    g_valEpoch_loss = checkpoint['valEpoch_loss']
    plot_loss(g_trainingEpoch_loss, g_valEpoch_loss, 'logs/LikeUNet/run_1', 'LikeUNet')
        

def load_csv_plot_loss_miou():

    col_names = ['Wall time', 'Step', 'Value']

    def get_data(csv_file):
        data = pd.read_csv(csv_file, skiprows=1, header=None, names=col_names)
        data.head(70)
        info = data.iloc[:, 2].values
        return info.tolist()
    
    # csv_file1 = 'logs/VAEUNet/run_1/csv/trainepoch_kl_loss.csv'
    # csv_file2 = 'logs/VAEUNet/run_1/csv/valepoch_kl_loss.csv'
    # data1 = get_data(csv_file1)
    # data2 = get_data(csv_file2)
    # plot_loss(data1, data2, 'logs/VAEUNet/run_1', 'VAEUNet', 'kl')

    # csv_file1 = 'logs/VAEUNet/run_0/csv/trainepoch_seg_loss.csv'
    # csv_file2 = 'logs/VAEUNet/run_0/csv/valepoch_seg_loss.csv'

    # data1 = get_data(csv_file1)
    # data2 = get_data(csv_file2)
    # plot_loss(data1, data2, 'logs/VAEUNet/run_0', 'VAEUNet', 'seg')
    
    # csv_file1 = 'logs/VAEUNet/run_0/csv/trainepoch_rec_loss.csv'
    # csv_file2 = 'logs/VAEUNet/run_0/csv/valepoch_rec_loss.csv'

    # data1 = get_data(csv_file1)
    # data2 = get_data(csv_file2)
    # plot_loss(data1, data2, 'logs/VAEUNet/run_0', 'VAEUNet', 'rec')

    # csv_file1 = 'logs/VAEUNet/run_0/csv/trainepoch_loss.csv'
    # csv_file2 = 'logs/VAEUNet/run_0/csv/valepoch_loss.csv'

    # data1 = get_data(csv_file1)
    # data2 = get_data(csv_file2)
    # plot_loss(data1, data2, 'logs/VAEUNet/run_0', 'VAEUNet', 'sum')

    # csv_file1 = 'logs/VAEUNet/run_0/csv/valf1_score.csv'
    # csv_file2 = 'logs/VAEUNet/run_0/csv/valmIoU.csv'

    # data1 = get_data(csv_file1)
    # data2 = get_data(csv_file2)
    # plot_mIou_f1score(data1, data2, 'logs/VAEUNet/run_0', 'VAEUNet')

    csv_file1 = 'logs/LikeUNet/run_2/csv/valf1_score.csv'
    csv_file2 = 'logs/LikeUNet/run_2/csv/valmIoU.csv'

    data1 = get_data(csv_file1)
    data2 = get_data(csv_file2)
    plot_mIou_f1score(data1, data2, 'logs/LikeUNet/run_2', 'VAEUNet')
    
def compose_img_with_mask():
    imgfilepath = 'in000616.jpg'
    maskfilepath = 'gt000616.png'
    img = cv2.imread(imgfilepath) 
    label = cv2.imread(maskfilepath)
    mask = label > 0
    # mask = np.expand_dims(mask, axis=2)
    # mask = np.repeat(mask, (1, 1, 3))
    img = img*mask
    cv2.imwrite('in000616_foreground.jpg', img)

if __name__ == "__main__":

    txt_filepath = 'D:/hongRui/GMU_course/CS782/Project/src/data/CDnet2014/badWeather/blizzard/temporalROI.txt'
    # print(get_temporalROI_from_txt(txt_filepath))

    a = '1'
    b = a.rjust(6, '0')
    print('x'+b)

    a = 'D:/hongRui/GMU_course/CS782/Project/src/data/CDnet2014/data\\turbulence\\turbulence3\\input\\in002192.jpg'
    # b = a.split('/') #['D:', 'hongRui', 'GMU_course', 'CS782', 'Project', 'src', 'data', 'CDnet2014', 'data\\turbulence\\turbulence3\\input\\in002192.jpg']
    # b = a.split('\\') ##['D:/hongRui/GMU_course/CS782/Project/src/data/CDnet2014/data', 'turbulence', 'turbulence3', 'input', 'in002192.jpg']
    # b = a.replace('\\', '/').split('/') #['D:', 'hongRui', 'GMU_course', 'CS782', 'Project', 'src', 'data', 'CDnet2014', 'data', 'turbulence', 'turbulence3', 'input', 'in002192.jpg']
    # print(b)
    a = np.random.random((4,2,6,6))
    b = np.argmax(a, axis=1)
    # print(b.shape, b)

    # a = np.random.random((2,1,3,3))
    # a[a>=0.5] = 1
    # a[a<0.5] = 0
    # print(a.shape, a)
    # a = a.squeeze()
    # print(a.shape, a)
    # load_weight_plot_loss()
    # a = torch.tensor([[1, 2], [2, 3]])
    # b = torch.tensor([[[1,2],[2,3]],[[-1,-2],[-2,-3]]])
    # print(b.shape)
    # x = torch.randn((2,1,3)) 
    # print(x)
    # x= x.repeat(1, 2, 1)
    # print(x)
    # load_csv_plot_loss_miou()
    compose_img_with_mask()