import os
import numpy as np
import scipy.misc as m
from PIL import Image
from torch.utils import data
from torchvision import transforms

# from dataloaders import joint_transforms as jo_trans
# from dataloaders.synthesis.synthesize_sample import AddNegSample as RandomAddNegSample

import logging
from os import listdir
from os.path import splitext
import torch
from torch.utils.data import Dataset
import cv2
import sys
import albumentations as albu
import random

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.), args = None):
        self.mean = mean
        self.std = std
        self.args = args
        
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        
        # img_name = sample['img_name']
        img = np.array(img).astype(np.float32)
        if isinstance(mask, Image.Image):
            mask = np.array(mask).astype(np.float32)

        img /= 255.0
        img -= self.mean
        img /= self.std
        # if img_name:
        #     return {'image': img,
        #         'label': mask,
        #         'img_name':img_name}
        # else:
        #     return {'image': img,
        #         'label': mask}
        if self.args and self.args.with_background:
            background = sample['background']
            background = np.array(background).astype(np.float32)
            background /= 255.0
            background -= self.mean
            background /= self.std
            sample['background'] = background

        sample['image'] = img
        sample['label'] = mask
        
        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, args = None):
        self.args = args
        
    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['label']
        # img_name = sample['img_name']
                
        img = np.array(img)
        if img.ndim == 3:
           img = img.astype(np.float32).transpose((2, 0, 1))
        else:
            img = np.expand_dims(img, axis=0)
        img = torch.from_numpy(img).float()
        if isinstance(mask, Image.Image):
            mask = np.array(mask).astype(np.float32)
        mask = torch.from_numpy(mask).float()
        if self.args and self.args.with_background:
            background = sample['background']
            background = np.array(background)
            if background.ndim == 3:
                background = background.astype(np.float32).transpose((2, 0, 1))
            else:
                background = np.expand_dims(background, axis=0)
            background = torch.from_numpy(background).float()
            sample['background'] = background
        # if img_name:
        #     return {'image': img,
        #         'label': mask,
        #         'img_name':img_name}
        # else:
        #     return {'image': img,
        #         'label': mask}
        sample['image'] = img
        sample['label'] = mask
        

        return sample


class FixedResize(object):
    def __init__(self, size, args = None):
        self.size = (size, size)  # size: (h, w)
        self.args = args

    def __call__(self, sample):
        # if self.args.testValTrain <= 1:
        #     return sample
        img = sample['image']
        mask = sample['label']
        
        # img_name = sample['img_name']
        if isinstance(mask, Image.Image):
            assert img.size == mask.size
            mask = mask.resize(self.size, Image.NEAREST)
        img = img.resize(self.size, Image.BILINEAR)

        if self.args and self.args.with_background:
            background = sample['background']
            background = background.resize(self.size, Image.BILINEAR)
            sample['background'] = background
        # if img_name:
        #     return {'image': img,
        #         'label': mask,
        #         'img_name':img_name}
        # else:
        #     return {'image': img,
        #         'label': mask}
        sample['image'] = img
        sample['label'] = mask
        
        return sample