import argparse
import os
import numpy as np
import math
from matplotlib import pyplot as plt

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from scipy.special import softmax

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
from model import bscGeneator, bscDiscriminator, LikeUNet, VAEUNet
import glob
from dataloader import CustomDataset, transfer_tensor_to_bgr_image
# os.makedirs("images", exist_ok=True)

import logging
from lr_scheduler import LR_Scheduler
from utils import *
from tqdm import tqdm
from loss import SegmentationLosses


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class bscGanWorker(object):
    def __init__(self, args):
        self.args = args

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("running on the GPU")
            self.cuda = True
        else:
            self.device = torch.device("cpu")
            print("running on the CPU")

        args.resume = args.resume.replace('\\', '/')

        
        exp_dir = args.resume[:args.resume.find(args.resume.split('/')[-1])]
        os.makedirs(exp_dir, exist_ok=True)

        self.logfile = os.path.join(exp_dir, 'inference_log.txt')
        if os.path.exists(self.logfile):
            os.remove(self.logfile)
        p=vars(self.args)
        log_file = open(self.logfile, "a+")
        for key, val in p.items():
            log_file.write(key + ':' + str(val) + '\n')
        log_file.write('\n')

        output_img_dir = os.path.join(exp_dir, 'result')
        os.makedirs(output_img_dir, exist_ok=True)
        
        self.exp_dir = exp_dir
        self.output_img_dir = output_img_dir
        if args.with_background:
            cin = 6
        else:
            cin = 3


        
        self.generator = bscGeneator()


        self.generator.to(self.device)

        testset = CustomDataset(args, split="test")


        self.testloader = DataLoader(testset, batch_size=args.batch_size, drop_last=True,
                                                shuffle=False, num_workers=2)
        


        self.evaluator = EvaluatorForeground(2)

        g_weight_filepath = args.resume
        fine_tune = False
        if g_weight_filepath and os.path.exists(g_weight_filepath):
            print(f"resume from {g_weight_filepath}")
            checkpoint = torch.load(g_weight_filepath, map_location=torch.device('cpu'))
            self.generator.load_state_dict(checkpoint['state_dict'])

        

        
    def write_log_to_txt(self, data):
        assert isinstance(data, str)
        log_file = open(self.logfile, "a+")
        if data.endswith('\n'):
            log_file.write(data)
        else:
            log_file.write(data+'\n')
        log_file.close()# 


    def inference(self, epoch = 0):
        

        self.generator.eval()

        self.evaluator.reset()
        tbar = tqdm(self.testloader)
        self.threshold = 0.15   #mIoU: 0.038900, f1_score: 0.069600
        self.threshold = 0.12   #mIoU: 0.108700, f1_score: 0.164000
        self.threshold = 0.115  #mIoU: 0.120300, f1_score: 0.176800
        self.threshold = 0.11   #mIoU: 0.133200, f1_score: 0.190300
        self.threshold = 0.105  #mIoU: 0.123800, f1_score: 0.180600
        self.threshold = 0.1    #mIoU: 0.116100, f1_score: 0.172300
        self.threshold = 0.09   #mIoU: 0.113900, f1_score: 0.169800
        self.threshold = 0.07   #mIoU: 0.101500, f1_score: 0.155600
        self.threshold = 0.05   #mIoU: 0.090300, f1_score: 0.142100
        

        for iter, sample in enumerate(tbar):
            if self.args.with_background:
                image_f, label, background, img_names = sample['image'], sample['label'], sample['background'], sample['img_name']
                image = torch.cat((image_f, background), 1)
                image = image.to(self.device)
                image_f = image_f.to(self.device)
            else:
                image_f, label, img_names = sample['image'], sample['label'], sample['img_name']
                image_f = image_f.to(self.device)
                image = image_f
            
            # Loss measures generator's ability to fool the discriminator
            with torch.no_grad():
                gen_output = self.generator(image)


            pred = gen_output.data.cpu().numpy()
            pred = pred.squeeze()
            ori_pre = pred.copy()
            ori_pre = (ori_pre*255).astype(np.uint8)

            target = label.cpu().numpy()
            pred[pred>=self.threshold] = 1
            pred[pred<self.threshold] = 0

            

            self.evaluator.add_batch(target.astype(int), pred.astype(int))

            groundtruth = np.expand_dims(target.astype(int)[0], axis=2)*255
            groundtruth = np.repeat(groundtruth, 3, axis=2)

            segmentation = np.expand_dims(ori_pre[0], axis=2)
            segmentation = np.repeat(segmentation, 3, axis=2)
            # img_f = put_text_on_img(img_f, 'origin image')
            # img_rec_f = put_text_on_img(img_rec_f, 'recovery image')
            # groundtruth = put_text_on_img(groundtruth, 'groundtruth')
            # segmentation = put_text_on_img(segmentation, 'segmentation')
            img_f = transfer_tensor_to_bgr_image(image_f)
            output = np.concatenate((img_f, groundtruth, segmentation), axis=1)
            output_filepath = os.path.join(self.output_img_dir, img_names[0])
            cv2.imwrite(output_filepath, output)


            tbar.set_description(f"Epoch {epoch:3d} {'infer'.ljust(6, ' ')} | iter {iter:04d}")


            if self.args.debug and iter > 50:
                break

        mIoU = self.evaluator.Mean_Intersection_over_Union()
        f1_score = self.evaluator.F1_Score()

        self.write_log_to_txt(f"Epoch {epoch} {'val'.ljust(6, ' ')} | mIoU: {mIoU:4f}, f1_score: {f1_score:4f}")
        print(f"Epoch {epoch} {'infer'.ljust(6, ' ')} | mIoU: {mIoU:4f}, f1_score: {f1_score:4f}")
        
        
        


class LikeUNetWorker(object):
    def __init__(self, args):
        self.args = args

        
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("running on the GPU")
            self.cuda = True
        else:
            self.device = torch.device("cpu")
            print("running on the CPU")

        args.resume = args.resume.replace('\\', '/')
        self.model_name = 'LikeUNet'
        self.threshold = 0.5
        exp_dir = args.resume[:args.resume.find(args.resume.split('/')[-1])]
        os.makedirs(exp_dir, exist_ok=True)

        self.logfile = os.path.join(exp_dir, 'inference_log.txt')
        if os.path.exists(self.logfile):
            os.remove(self.logfile)
        p=vars(self.args)
        log_file = open(self.logfile, "a+")
        for key, val in p.items():
            log_file.write(key + ':' + str(val) + '\n')
        log_file.write('\n')

        output_img_dir = os.path.join(exp_dir, 'result')
        os.makedirs(output_img_dir, exist_ok=True)
        
        self.exp_dir = exp_dir
        self.output_img_dir = output_img_dir
        if args.with_background:
            cin = 6
        else:
            cin = 3

        self.model = LikeUNet(cin)

        self.model.to(self.device)

        # # Initialize weights
        # self.model.apply(weights_init_normal)
        testset = CustomDataset(args, split="test")
        # print('trainset', trainset)
        self.testloader = DataLoader(testset, batch_size=args.batch_size, drop_last=True,
                                                shuffle=False, num_workers=2)
        


        self.evaluator = EvaluatorForeground(2)

      
        self.epoch_f1_score = []
        self.epoch_mIou = []
        g_weight_filepath = args.resume
        if g_weight_filepath and os.path.exists(g_weight_filepath):
            print(f"resume from {g_weight_filepath}")
            checkpoint = torch.load(g_weight_filepath, map_location=torch.device('cpu'))
            self.model.load_state_dict(checkpoint['state_dict'])



    def write_log_to_txt(self, data):
        assert isinstance(data, str)
        log_file = open(self.logfile, "a+")
        if data.endswith('\n'):
            log_file.write(data)
        else:
            log_file.write(data+'\n')
        log_file.close()# 

    def inference(self, epoch = 0):
        

        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.testloader)
        # if self.args.master:
        #     print(f'rank {self.args.rank} num_img_tr: {num_img_tr}')


        for iter, sample in enumerate(tbar):
            # print(f'rank {self.args.rank} dataload time {round(time.time() - start, 3)}')
            # start = time.time()
            
            # print('target', target.size(), image.size())
            if self.args.with_background:
                image_f, label, background, img_names = sample['image'], sample['label'], sample['background'], sample['img_name']
                image = torch.cat((image_f, background), 1)
                image = image.to(self.device)
                image_f = image_f.to(self.device)
            else:
                image_f, label, img_names = sample['image'], sample['label'], sample['img_name']
                image_f = image_f.to(self.device)
                image = image_f
            
            label = label.to(self.device)


        
            # Loss measures generator's ability to fool the discriminator
            with torch.no_grad():
                seg = self.model(image)

            pred = seg.data.cpu().numpy()
            target = label.cpu().numpy()
            pred = softmax(pred, axis=1)
            pred = np.argmax(pred, axis=1)
            self.evaluator.add_batch(target.astype(int), pred.astype(int))

            img_f = transfer_tensor_to_bgr_image(image_f)
            # print('img_f', img_f.shape, img_f.min(), img_f.max())
            # 

            groundtruth = np.expand_dims(target.astype(int)[0], axis=2)*255
            groundtruth = np.repeat(groundtruth, 3, axis=2)

            segmentation = np.expand_dims(pred.astype(int)[0], axis=2)*255
            segmentation = np.repeat(segmentation, 3, axis=2)
            
            # img_f = put_text_on_img(img_f, 'origin image')
            # img_rec_f = put_text_on_img(img_rec_f, 'recovery image')
            # groundtruth = put_text_on_img(groundtruth, 'groundtruth')
            # segmentation = put_text_on_img(segmentation, 'segmentation')
            
            output = np.concatenate((img_f, groundtruth, segmentation), axis=1)
            output_filepath = os.path.join(self.output_img_dir, img_names[0])
            cv2.imwrite(output_filepath, output)

            tbar.set_description(f"Epoch {epoch:3d} {'infer'.ljust(6, ' ')} | iter {iter:04d}")

            if self.args.debug and iter > 50:
                break



        mIoU = self.evaluator.Mean_Intersection_over_Union()
        f1_score = self.evaluator.F1_Score()

        print(f"mIoU: {mIoU:4f}, f1_score: {f1_score:4f}")
        self.write_log_to_txt(f"mIoU: {mIoU:4f}, f1_score: {f1_score:4f}")



class VAEUNetWorker(object):
    def __init__(self, args):
        self.args = args

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("running on the GPU")
            self.cuda = True
        else:
            self.device = torch.device("cpu")
            print("running on the CPU")

        args.resume = args.resume.replace('\\', '/')
        self.model_name = 'VAEUNet'
        self.threshold = 0.5
        exp_dir = args.resume[:args.resume.find(args.resume.split('/')[-1])]
        os.makedirs(exp_dir, exist_ok=True)

        self.logfile = os.path.join(exp_dir, 'inference_log.txt')
        if os.path.exists(self.logfile):
            os.remove(self.logfile)
        p=vars(self.args)
        log_file = open(self.logfile, "a+")
        for key, val in p.items():
            log_file.write(key + ':' + str(val) + '\n')
        log_file.write('\n')

        output_img_dir = os.path.join(exp_dir, 'result')
        os.makedirs(output_img_dir, exist_ok=True)
        
        self.exp_dir = exp_dir
        self.output_img_dir = output_img_dir
        if args.with_background:
            cin = 6
        else:
            cin = 3
        self.model = VAEUNet(cin)

        self.model.to(self.device)

        # # Initialize weights
        # self.model.apply(weights_init_normal)
        testset = CustomDataset(args, split="test")
        # print('trainset', trainset)
        self.testloader = DataLoader(testset, batch_size=args.batch_size, drop_last=True,
                                                shuffle=False, num_workers=2)
        
        self.evaluator = EvaluatorForeground(2)

        self.epoch_f1_score = []
        self.epoch_mIou = []
        g_weight_filepath = args.resume
        if g_weight_filepath and os.path.exists(g_weight_filepath):
            print(f"resume from {g_weight_filepath}")
            checkpoint = torch.load(g_weight_filepath, map_location=torch.device('cpu'))
            self.model.load_state_dict(checkpoint['state_dict'])
            
        
    def write_log_to_txt(self, data):
        assert isinstance(data, str)
        log_file = open(self.logfile, "a+")
        if data.endswith('\n'):
            log_file.write(data)
        else:
            log_file.write(data+'\n')
        log_file.close()# 


    def inference(self, epoch = 0):        
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.testloader)
        # if self.args.master:
        #     print(f'rank {self.args.rank} num_img_tr: {num_img_tr}')

        for iter, sample in enumerate(tbar):
            # print(f'rank {self.args.rank} dataload time {round(time.time() - start, 3)}')
            # start = time.time()
            
            # print('target', target.size(), image.size())
            if self.args.with_background:
                image_f, label, background, img_names = sample['image'], sample['label'], sample['background'], sample['img_name']
                image = torch.cat((image_f, background), 1)
                image = image.to(self.device)
                image_f = image_f.to(self.device)
            else:
                image_f, label, img_names = sample['image'], sample['label'], sample['img_name']
                image_f = image_f.to(self.device)
                image = image_f
            
            label = label.to(self.device)

        
            # Loss measures generator's ability to fool the discriminator
            with torch.no_grad():
                seg, p_x, mu, logvar = self.model(image)

            pred = seg.data.cpu().numpy()
            target = label.cpu().numpy()
            pred = softmax(pred, axis=1)
            pred = np.argmax(pred, axis=1)
            self.evaluator.add_batch(target.astype(int), pred.astype(int))

            img_f = transfer_tensor_to_bgr_image(image_f)
            # print('img_f', img_f.shape, img_f.min(), img_f.max())
            # 
            img_rec_f = transfer_tensor_to_bgr_image(p_x)
            # 

            groundtruth = np.expand_dims(target.astype(int)[0], axis=2)*255
            groundtruth = np.repeat(groundtruth, 3, axis=2)

            segmentation = np.expand_dims(pred.astype(int)[0], axis=2)*255
            segmentation = np.repeat(segmentation, 3, axis=2)
            
            # img_f = put_text_on_img(img_f, 'origin image')
            # img_rec_f = put_text_on_img(img_rec_f, 'recovery image')
            # groundtruth = put_text_on_img(groundtruth, 'groundtruth')
            # segmentation = put_text_on_img(segmentation, 'segmentation')
            
            imgs = np.concatenate((img_f, img_rec_f), axis=1)
            segs = np.concatenate((groundtruth, segmentation), axis=1)
            output = np.concatenate((imgs, segs), axis=0)
            output_filepath = os.path.join(self.output_img_dir, img_names[0])
            cv2.imwrite(output_filepath, output)

            tbar.set_description(f"Epoch {epoch:3d} {'infer'.ljust(6, ' ')} | iter {iter:04d}")

            if self.args.debug and iter > 50:
                break

        mIoU = self.evaluator.Mean_Intersection_over_Union()
        f1_score = self.evaluator.F1_Score()

        print(f"mIoU: {mIoU:4f}, f1_score: {f1_score:4f}")
        self.write_log_to_txt(f"mIoU: {mIoU:4f}, f1_score: {f1_score:4f}")
       
    

            
def main(args):
    if args.model_name == 'bscGAN':
        worker = bscGanWorker(args)
        worker.inference()
    elif args.model_name == 'LikeUNet':
        worker = LikeUNetWorker(args)
        worker.inference()
    elif args.model_name == 'VAEUNet':
        worker = VAEUNetWorker(args)
        worker.inference()

if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")

    parser.add_argument("--img_size", type=int, default=256, help="size of each image dimension")
    parser.add_argument("--ignore_index", type=int, default=255, help="ignore_index")
    parser.add_argument('--with_background', action='store_true', default=False, #True, False
                    help='with_background image')
    parser.add_argument('--dataset_dir', type=str, default='data/CDnet2014', help='dataset dir')
    parser.add_argument('--resume', type=str, default=None, help='resume')
    parser.add_argument('--resume2', type=str, default=None, help='resume')

    parser.add_argument('--debug', action='store_true', default=True, #True, False
                    help='debug mode')
    # parser.add_argument('--model_name', type=str, default='bscGAN', help='model_name')
    # parser.add_argument('--model_name', type=str, default='LikeUNet', help='model_name')
    parser.add_argument('--model_name', type=str, default='VAEUNet', help='model_name')
    
    opt = parser.parse_args()

    main(opt)
    