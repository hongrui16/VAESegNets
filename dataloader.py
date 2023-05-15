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


from utils import *
from transform_fun import *



class CustomDataset(Dataset):

    def __init__(self, args, split="train"):
        self.args = args
        self.root = args.dataset_dir
        self.split = split
        self.ignore_index = args.ignore_index
        self.with_background = args.with_background
        self.dataset_name = args.dataset_name
        # print('args.ignore_index', args.ignore_index)
    
        txt_filepath = os.path.join(self.root, f'{self.split}.txt')
        self.img_filepaths = read_txt_to_list(txt_filepath)
        # print('self.img_filepaths ', len(self.img_filepaths) )

        if self.dataset_name == 'cdnet':
            self.img_filepaths = self.img_filepaths[:6000]
            
        elif self.dataset_name == 'bmc':
            self.img_filepaths = self.img_filepaths[:3200]
            if split == 'test':
                temp_list = []
                for i, line in enumerate(self.img_filepaths):
                    if i %2 == 0:
                        temp_list.append(line)
                self.img_filepaths = temp_list
        if split == 'train':
            random.shuffle(self.img_filepaths)
        


    def __len__(self):
        # return len(self.img_ids)
        return len(self.img_filepaths)

    def __getitem__(self, index):
        img_path = self.img_filepaths[index]
        img_path = img_path.replace('\\', '/')
        
        img_name = img_path.split('/')[-1]
        img_dir = img_path[:img_path.find(img_name)]

        if self.dataset_name == 'cdnet':
            lbl_name = img_name.replace('.jpg', '.png')
            lbl_name = lbl_name.replace('in', 'gt')
            
            lbl_dir = img_dir.replace('input', 'groundtruth')
            # lbl_path = img_path.replace('image', 'label').replace('.jpg', '.png')
            lbl_path = os.path.join(lbl_dir, lbl_name)
            img = cv2.imread(img_path)
            # _img = Image.open(img_path).convert('RGB')
            
            
            label = cv2.imread(lbl_path, 0).astype(np.uint8)

            

            ROI_filepath = os.path.join(img_dir[:img_dir.find('input')], 'ROI.bmp')            
            roi = cv2.imread(ROI_filepath, 0)
            h, w = roi.shape 
            img = cv2.resize(img, (w,h))
            label = cv2.resize(label, (w,h), interpolation=cv2.INTER_NEAREST)
            roi_mask = roi==255
            mask_3d = np.repeat(roi_mask[:, :, np.newaxis], 3, axis=2)
            img *= mask_3d
            label *= roi_mask

            label, img = self.encode_segmap(label, img)
            # print('_img.shape', _img.shape)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)

                
            label = Image.fromarray(label)

            if self.with_background:
                background_filepath = os.path.join(img_dir[:img_dir.find('input')], 'background.jpg')
                if not os.path.exists(background_filepath):
                    background_filepath = os.path.join(img_dir, 'in000001.jpg')
                # print('background_filepath', background_filepath)
                background = cv2.imread(background_filepath)
                background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
                background = Image.fromarray(background)
                sample = {'image': img, 'label': label, 'background': background, 'img_name': img_name}
            else:
                sample = {'image': img, 'label': label, 'img_name': img_name}

        elif self.dataset_name == 'bmc':
            lbl_name = img_name.replace('.jpg', '.png')
            
            lbl_path = img_path.replace('image', 'groundtruth').replace('.jpg', '.png')
            # lbl_path = img_path.replace('image', 'label').replace('.jpg', '.png')
            # print(lbl_path)
            img = cv2.imread(img_path)
            # _img = Image.open(img_path).convert('RGB')
            
            
            label = cv2.imread(lbl_path, 0).astype(np.uint8)

        
            label, img = self.encode_segmap(label, img)
            # print('_img.shape', _img.shape)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)

            h,w = label.shape
            label = Image.fromarray(label)

            if self.with_background:
                background_filepath = os.path.join(img_dir[:img_dir.find('image')], 'background.jpg')
                if not os.path.exists(background_filepath):
                    background = np.zeros((h,w, 3), np.uint8)
                # print('background_filepath', background_filepath)
                else:
                    background = cv2.imread(background_filepath)
                    background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
                # print('background_filepath', background_filepath)
                background = Image.fromarray(background)
                sample = {'image': img, 'label': label, 'background': background, 'img_name': img_name}
            else:
                sample = {'image': img, 'label': label, 'img_name': img_name}

        
        return self.transform_sample(sample)

        # if self.split == 'train':
        #     return self.transform_train(sample)
        # elif self.split == 'val':
        #     return self.transform_val(sample)
        # elif self.split == 'test':
        #     return self.transform_test(sample)

    def encode_segmap(self, label, img = None):
        '''
        CDW-2014/CDnet
        a sub-directory named "input" containing a separate JPEG file for each frame of the input video
        a sub-directory named "groundtruth" containing a separate BMP file for each frame of the groundtruth
        "an empty folder named "results" for binary results (1 binary image per frame per video you have processed)
        files named "ROI.bmp" and "ROI.jpg" showing the spatial region of interest
        a file named "temporalROI.txt" containing two frame numbers. Only the frames in this range will be used to calculate your score
        0 : Static
        50 : Hard shadow
        85 : Outside region of interest
        170 : Unknown motion (usually around moving objects, due to semi-transparency and motion blur)
        255 : Motion
        '''        
        if self.dataset_name == 'cdnet':
            if label.any() > 0:
                mask_bk = label.copy()
                
                label[mask_bk == 0] = 0
                label[mask_bk == 50] = 1
                label[mask_bk == 170] = 0
                label[mask_bk == 255] = 1
                label[mask_bk==85] = self.args.ignore_index #255
                
            img = img.astype(np.uint8)
            mask = mask_bk!=85
            mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
            img *= mask_3d
        elif self.dataset_name == 'bmc':
            if label.any() > 0:
                mask_bk = label.copy()
                label[mask_bk == 0] = 0
                label[mask_bk == 255] = 1
                
        return label, img.astype(np.uint8)

    def transform_sample(self, sample):
        composed_transforms = transforms.Compose([

            FixedResize(size=self.args.img_size, args = self.args),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), args = self.args),
            ToTensor(args = self.args)])

        return composed_transforms(sample)
    
    def transform_train(self, sample):
        composed_transforms = transforms.Compose([

            FixedResize(size=self.args.img_size),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([

            FixedResize(size=self.args.img_size),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensor()])

        return composed_transforms(sample)

    def transform_test(self, sample):
        composed_transforms = transforms.Compose([

            FixedResize(size=self.args.img_size),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensor()])

        return composed_transforms(sample)
        
def transfer_tensor_to_bgr_image(img_tensor):
    img_tmp = np.transpose(img_tensor.detach().cpu().numpy()[0], axes=[1, 2, 0])
    img_tmp *= (0.229, 0.224, 0.225)
    img_tmp += (0.485, 0.456, 0.406)
    min_value = img_tmp.min()
    max_value = img_tmp.max()
    img_tmp = (img_tmp-min_value)/(max_value-min_value)
    img_tmp *= 255.0
    img_bgr = img_tmp[:,:,::-1]
    img_bgr = img_bgr.astype(np.uint8)
    return img_bgr

if __name__ == '__main__':
    # from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    parser = argparse.ArgumentParser()
    parser.add_argument('-im', '--input_dir', type=str, default=None)
    parser.add_argument('-om', '--output_dir', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=16,
                    metavar='N', help='input batch size for \
                            training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=8,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
   
    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    root = args.input_dir

    # basicDataset_train = BasicDataset(args, root, split="train")
    # basicDataset_test = BasicDataset(args, root, split="test")

    # train_loader = DataLoader(basicDataset_train, batch_size=2, shuffle=False, num_workers=2)
    # test_loader = DataLoader(basicDataset_test, batch_size=2, shuffle=False, num_workers=2)

    # def save_img_mask(loader):
    #     for ii, sample in enumerate(loader):
    #         if ii == 3:
    #             break
    #         batch_size = sample["image"].size()[0]
    #         # print('batch_size: ', batch_size)
    #         for jj in range(batch_size):

    #             img = sample['image'].numpy()
    #             gt = sample['label'].numpy()
    #             img_name =  sample['img_name']
    #             img_name_perfix = img_name.split('.')[0]
    #             segmap = np.array(gt[jj]).astype(np.uint8)
    #             # segmap = decode_segmap(segmap, dataset='cityscapes')
    #             img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
    #             img_tmp *= (0.229, 0.224, 0.225)
    #             img_tmp += (0.485, 0.456, 0.406)
    #             img_tmp *= 255.0
    #             img_tmp = img_tmp.astype(np.uint8)
                
    #             # plt.figure()
    #             # plt.title('display')
    #             # plt.subplot(211)
    #             # plt.imshow(img_tmp)
    #             # plt.subplot(212)
    #             # plt.imshow(segmap)
    #             # ax = plt.subplot(4, batch_size*2, ii*batch_size*2 + 2*jj+1), plt.imshow(img_tmp), plt.title(f'img_{ii*batch_size + jj}'), plt.xticks([]), plt.yticks([])
    #             # ax = plt.subplot(4, batch_size*2, ii*batch_size*2 + 2*jj+2), plt.imshow(segmap*60), plt.title(f'mask_{ii*batch_size + jj}'), plt.xticks([]), plt.yticks([])
    #             # if segmap.ndim == 2:
    #             #     plt.gray()

    #             cv2.imwrite(os.path.join(output_dir, f'{img_name_perfix}.jpg'), img_tmp)
    #             cv2.imwrite(os.path.join(output_dir, f'{img_name_perfix}.png'), segmap*60)

    # save_img_mask(train_loader)
    # save_img_mask(test_loader)
    # # plt.show()
    # # plt.show(block=True)

