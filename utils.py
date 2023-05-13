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

def write_list_to_txt(txt_filepath, lists):
    f = open(txt_filepath, "a+")
    for l in lists:
        f.write(str(l)+'\n')
    f.close()#

def read_txt_to_list(txt_filepath):
    lists = []
    with open(txt_filepath) as f:
        lines = f.readlines()
        for line in lines:
            lists.append(line.strip('\n'))
    # print(len(lists))
    return lists

def get_temporalROI_from_txt(txt_filepath):

    with open(txt_filepath) as f:
        lines = f.readlines()
        
        line = lines[0].strip('\n')
        # print(line.split(' '))
        min_n, max_n = line.split(' ')
        min_n = int(min_n)
        max_n = int(max_n)
        f.close()
    return min_n, max_n
    # print(len(lists))
    # return lists

def compute_foregound_iou(y_pred, y_true):
    intersection = np.logical_and(y_pred, y_true)
    union = np.logical_or(y_pred, y_true)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


class EvaluatorForeground(object):
    def __init__(self, num_class, args = None):
        self.args = args
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        Acc = round(Acc, 4)
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        Acc = round(Acc, 4)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        # MIoU_ori = MIoU.copy()
        MIoU = MIoU[1:]
        MIoU = np.nanmean(MIoU)
        MIoU = round(MIoU, 4)
        # print('MIoU_ori', MIoU_ori, 'MIoU', MIoU)

        return MIoU
    
    def F1_Score(self):
        f1_score = 2*np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) +
                    2*np.diag(self.confusion_matrix))
        # MIoU_ori = MIoU.copy()
        f1_score = f1_score[1:]
        f1_score = np.nanmean(f1_score)
        f1_score = round(f1_score, 4)
        # print('MIoU_ori', MIoU_ori, 'MIoU', MIoU)

        return f1_score
    

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        FWIoU = round(FWIoU, 4)
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        # print('label.shape', label.shape)
        count = np.bincount(label, minlength=self.num_class**2)
        # print('count.shape', count, count.shape)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)



def plot_loss(trainingEpoch_loss, valEpoch_loss, exp_dir, name = '', postfix = ''):
    # fig = plt.figure()
    plt.plot(trainingEpoch_loss, f'training {postfix}loss')
    plt.plot(valEpoch_loss, f'val {postfix}loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(exp_dir, f"{name}_loss.jpg"))
    plt.close()


def plot_mIou_f1score(mIou, f1_score, exp_dir, name = '', postfix = ''):
    # fig = plt.figure()
    plt.plot(mIou, 'mIou')
    plt.plot(f1_score, 'f1_score')
    plt.xlabel("Epoch")
    plt.ylabel("value")
    plt.legend()
    plt.savefig(os.path.join(exp_dir, f"{name}_metrics.jpg"))
    plt.close()


def load_weight_plot_loss():
    g_weight_filepath = 'logs/VAEUNet/run_0/VAEUNet_checkpoint.pth.tar'
    checkpoint = torch.load(g_weight_filepath, map_location=torch.device('cpu'))

    g_trainingEpoch_loss = checkpoint['trainingEpoch_loss']
    g_valEpoch_loss = checkpoint['valEpoch_loss']
    plot_loss(g_trainingEpoch_loss, g_valEpoch_loss, 'logs/LikeUNet/run_1', 'LikeUNet')
        

def load_weight_plot_loss_miou():
    g_weight_filepath = 'logs/LikeUNet/run_1/LikeUNet_checkpoint.pth.tar'
    checkpoint = torch.load(g_weight_filepath, map_location=torch.device('cpu'))

    g_trainingEpoch_loss = checkpoint['trainingEpoch_loss']
    g_valEpoch_loss = checkpoint['valEpoch_loss']
    trainingEpoch_loss += checkpoint['trainingEpoch_loss']
    valEpoch_loss += checkpoint['valEpoch_loss']
    epoch_f1_score += checkpoint['epoch_f1_score']
    epoch_mIou += checkpoint['epoch_mIou']
    
    plot_loss(g_trainingEpoch_loss, g_valEpoch_loss, 'logs/LikeUNet/run_1', 'LikeUNet')


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
    a = torch.tensor([[1, 2], [2, 3]])
    b = torch.tensor([[[1,2],[2,3]],[[-1,-2],[-2,-3]]])
    print(b.shape)
    x = torch.randn((2,1,3)) 
    print(x)
    x= x.repeat(1, 2, 1)
    print(x)
