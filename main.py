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

        self.threshold = 0.5
        self.dataset_name = args.dataset_name
        log_dir = sorted(glob.glob(os.path.join('logs', 'bscGan', self.dataset_name, 'run_*')), key=lambda x: int(x.split('_')[-1]))
        run_id = int(log_dir[-1].split('_')[-1]) + 1 if log_dir else 0
        exp_dir = os.path.join('logs', 'bscGan', self.dataset_name, 'run_{}'.format(str(run_id)))
        print('exp_dir', exp_dir)
        os.makedirs(exp_dir, exist_ok=True)

        self.logfile = os.path.join(exp_dir, 'log.txt')
        p=vars(self.args)
        log_file = open(self.logfile, "a+")
        for key, val in p.items():
            log_file.write(key + ':' + str(val) + '\n')
        log_file.write('\n')

        output_img_dir = os.path.join(exp_dir, 'result')
        os.makedirs(output_img_dir, exist_ok=True)
        
        self.log_dir = log_dir
        self.exp_dir = exp_dir
        self.output_img_dir = output_img_dir
        
        self.generator = bscGeneator()
        self.discriminator = bscDiscriminator()

        self.generator.to(self.device)
        self.discriminator.to(self.device)

        # # Initialize weights
        self.generator.apply(weights_init_normal)
        self.discriminator.apply(weights_init_normal)


        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))


        self.adversarial_loss = torch.nn.BCELoss(reduce = True)
        self.mes_loss = nn.MSELoss()
        
        self.adversarial_loss = self.adversarial_loss.to(self.device)
        self.mes_loss = self.mes_loss.to(self.device)
        
        
        trainset = CustomDataset(args, split="train")
        testset = CustomDataset(args, split="test")
        # print('trainset', trainset)
        self.trainloader = DataLoader(trainset, batch_size=args.batch_size, drop_last=True, 
                                                shuffle=True, num_workers=2)
        self.testloader = DataLoader(testset, batch_size=args.batch_size, drop_last=True,
                                                shuffle=False, num_workers=2)
        
        self.writer = SummaryWriter(exp_dir)


        self.evaluator = EvaluatorForeground(2)

        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                                args.n_epochs, len(self.trainloader), args = args)
        
        self.best_pred = 0
        self.start_epoch = 0
        self.g_trainingEpoch_loss = []
        self.g_valEpoch_loss = []
        self.d_trainingEpoch_loss = []
        self.d_valEpoch_loss = []

        g_weight_filepath = args.resume
        fine_tune = False
        if g_weight_filepath and os.path.exists(g_weight_filepath):
            print(f"resume from {g_weight_filepath}")
            checkpoint = torch.load(g_weight_filepath, map_location=torch.device('cpu'))
            self.generator.load_state_dict(checkpoint['state_dict'])
            if not fine_tune:
                self.start_epoch = checkpoint['epoch']
                self.generator.load_state_dict(checkpoint['optimizer'])
                self.g_trainingEpoch_loss += checkpoint['trainingEpoch_loss']
                self.g_valEpoch_loss += checkpoint['valEpoch_loss']
                self.best_pred += checkpoint['best_pred']
        
        d_weight_filepath = args.resume2
        if d_weight_filepath and os.path.exists(d_weight_filepath):
            print(f"resume from {d_weight_filepath}")
            checkpoint = torch.load(d_weight_filepath, map_location=torch.device('cpu'))
            self.discriminator.load_state_dict(checkpoint['state_dict'])
            if not fine_tune:
                self.discriminator.load_state_dict(checkpoint['optimizer'])
                self.d_trainingEpoch_loss += checkpoint['trainingEpoch_loss']
                self.d_valEpoch_loss += checkpoint['valEpoch_loss']


        print(f"train dataset num: {len(self.trainloader)}, test dataset num: {len(self.testloader)}")
        print(f"Starting epoch {self.start_epoch}:\n")

        
                
    def save_checkpoint(self, state, is_best, model_name=''):
        """Saves checkpoint to disk"""
        best_model_filepath = os.path.join(self.exp_dir, f'{model_name}_model_best.pth.tar')
        filename = os.path.join(self.exp_dir, f'{model_name}_checkpoint.pth.tar')
        torch.save(state, filename)
        if is_best:   
            torch.save(state, best_model_filepath)

    def write_log_to_txt(self, data):
        assert isinstance(data, str)
        log_file = open(self.logfile, "a+")
        if data.endswith('\n'):
            log_file.write(data)
        else:
            log_file.write(data+'\n')
        log_file.close()# 

    def training(self, epoch):

        self.generator.train()
        self.discriminator.train()
        tbar = tqdm(self.trainloader)
        # if self.args.master:
        #     print(f'rank {self.args.rank} num_img_tr: {num_img_tr}')
        Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor

        step_g_loss = []
        step_d_loss = []

        # for iter, sample in enumerate(tbar):
        for iter, sample in enumerate(tbar):
            # print(f'rank {self.args.rank} dataload time {round(time.time() - start, 3)}')
            # start = time.time()
            image, label, background, img_name = sample['image'], sample['label'], sample['background'], sample['img_name']
            # print('target', target.size(), image.size())
            # print('image.shape', image.shape)
            # print('background.shape', background.shape)

            if self.args.with_background:
                image = torch.cat((image, background), 1)
            image, label = image.to(self.device), label.to(self.device)
            
            # print('image.shape', image.shape)

            valid = Variable(Tensor(image.size(0), 1).fill_(1.0), requires_grad=False) # imgs.size(0) == batch_size(1 batch) == 64, *TEST_CODE
            fake = Variable(Tensor(image.size(0), 1).fill_(0.0), requires_grad=False) # And Variable is for caclulate gradient. In fact, you can use it, but you don't have to. 

            # ------------
            # Train Generator
            # ------------
            # Generate a batch of images

            self.optimizer_G.zero_grad()

            gen_output = self.generator(image)

            # print('image.shape', image.shape) #torch.Size([16, 6, 256, 256])
            # print('gen_output.shape', gen_output.shape) #torch.Size([16, 1, 256, 256])

            # Loss measures generator's ability to fool the discriminator
            decision = self.discriminator(torch.cat((image, gen_output.detach()), 1))
            adver_loss = self.adversarial_loss(decision, valid)
            mes_loss = self.mes_loss(gen_output, label)
            g_loss = adver_loss + 10*mes_loss
            g_loss.backward()
            self.optimizer_G.step()
            self.scheduler(self.optimizer_G, iter, epoch)

            # ---------------------
            #  Train Discriminator
            # ---------------------

            self.optimizer_D.zero_grad()

            # print('image.shape', image.shape) #torch.Size([16, 6, 256, 256])
            # print('label.shape', label.shape)  #torch.Size([16, 256, 256])
            # print('label.shape', torch.unsqueeze(label, 1).shape) #torch.Size([16, 1, 256, 256])
             # Measure discriminator's ability to classify real from generated samples
            # decision_1 = self.discriminator(torch.cat((image, torch.unsqueeze(label, 1)), 1))
            # print('decision_1',decision_1)
            # print('decision_1.shape', decision_1.shape) #torch.Size([16, 1])
            # print('valid.shape', valid.shape) #torch.Size([16, 1])
            # real_loss = self.adversarial_loss(self.discriminator(torch.cat((image, torch.unsqueeze(label, 1)), 1)), valid) # torch.nn.BCELoss() compare result(64x1) and valid(64x1, filled with 1)
            # fake_loss = self.adversarial_loss(self.discriminator(torch.cat((image, gen_output.detach()), 1)), fake) # We are learning the discriminator now. So have to use detach() 
            decision_t = self.discriminator(torch.cat((image, torch.unsqueeze(label, 1)), 1))
            decision_f = self.discriminator(torch.cat((image, gen_output.detach()), 1))
            real_loss = self.adversarial_loss(decision_t, valid) # torch.nn.BCELoss() compare result(64x1) and valid(64x1, filled with 1)
            fake_loss = self.adversarial_loss(decision_f, fake) # We are learning the discriminator now. So have to use detach() 

     
            # print('real_loss', real_loss, 'fake_loss', fake_loss)                                                   
            d_loss = (real_loss + fake_loss) / 2
            
            d_loss.backward()# If didn't use detach() for gen_imgs, all weights of the generator will be calculated with backward(). 
            self.optimizer_D.step()
            
            self.scheduler(self.optimizer_D, iter, epoch)
            tbar.set_description(f"Epoch {epoch:3d} {'train'.ljust(6, ' ')} | iter {iter:04d} g_loss: {g_loss.item():4f} d_loss: {d_loss.item():4f}")
            

            step_g_loss.append(g_loss.item())
            step_d_loss.append(d_loss.item())

            if self.args.debug and iter > 30:
                break

        epoch_g_loss = np.array(step_g_loss).mean()
        epoch_d_loss = np.array(step_d_loss).mean()
        self.writer.add_scalar('train/epoch_g_loss', epoch_g_loss, epoch)
        self.writer.add_scalar('train/epoch_d_loss', epoch_d_loss, epoch)
        self.write_log_to_txt(f"Epoch {epoch:3d} {'train'.ljust(6, ' ')} | epoch_g_loss: {epoch_g_loss:4f} epoch_d_loss: {epoch_d_loss:4f}")
        print(f"Epoch {epoch} {'train'.ljust(6, ' ')} | epoch_g_loss: {epoch_g_loss:4f} epoch_d_loss: {epoch_d_loss:4f}")
            # print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        self.g_trainingEpoch_loss.append(epoch_g_loss)
        self.d_trainingEpoch_loss.append(epoch_d_loss)



    def validation(self, epoch = 0):
        

        self.generator.eval()
        self.discriminator.eval()
        self.evaluator.reset()
        tbar = tqdm(self.testloader)
        # if self.args.master:
        #     print(f'rank {self.args.rank} num_img_tr: {num_img_tr}')
        Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
        is_best_epoch = False


        step_g_loss = []
        step_d_loss = []

        for iter, sample in enumerate(tbar):
            # print(f'rank {self.args.rank} dataload time {round(time.time() - start, 3)}')
            # start = time.time()
            image, label, background, img_name = sample['image'], sample['label'], sample['background'], sample['img_name']
            # print('target', target.size(), image.size())
            if self.args.with_background:
                image = torch.cat((image, background), 1)
            image, label = image.to(self.device), label.to(self.device)

            valid = Variable(Tensor(image.size(0), 1).fill_(1.0), requires_grad=False) # imgs.size(0) == batch_size(1 batch) == 64, *TEST_CODE
            fake = Variable(Tensor(image.size(0), 1).fill_(0.0), requires_grad=False) # And Variable is for caclulate gradient. In fact, you can use it, but you don't have to. 

            # ------------
            # Train Generator
            # ------------
            # Generate a batch of images

            self.optimizer_G.zero_grad()
        
            # Loss measures generator's ability to fool the discriminator
            with torch.no_grad():
                gen_output = self.generator(image)
                discri_output = self.discriminator(torch.cat((image, gen_output), 1))
            g_loss = self.adversarial_loss(discri_output, valid) + 10*self.mes_loss(gen_output, label)
            pred = gen_output.data.cpu().numpy()
            target = label.cpu().numpy()
            pred[pred>=self.threshold] = 1
            pred[pred<self.threshold] = 0
            pred = pred.squeeze()
            self.evaluator.add_batch(target.astype(int), pred.astype(int))


             # Measure discriminator's ability to classify real from generated samples
            with torch.no_grad():
                discri_output_v = self.discriminator(torch.cat((image, torch.unsqueeze(label, 1)), 1))
                discri_output_f = self.discriminator(torch.cat((image, gen_output.detach()), 1))
            real_loss = self.adversarial_loss(discri_output_v, valid) # torch.nn.BCELoss() compare result(64x1) and valid(64x1, filled with 1)
            fake_loss = self.adversarial_loss(discri_output_f, fake) # We are learning the discriminator now. So have to use detach() 
                                                                                
            d_loss = (real_loss + fake_loss) / 2
            

            tbar.set_description(f"Epoch {epoch:3d} {'val'.ljust(6, ' ')} | iter {iter:04d} g_loss: {g_loss.item():4f} d_loss: {d_loss.item():4f}")

            step_g_loss.append(g_loss.item())
            step_d_loss.append(d_loss.item())

            if self.args.debug and iter > 30:
                break

        mIoU = self.evaluator.Mean_Intersection_over_Union()
        f1_score = self.evaluator.F1_Score()
        epoch_g_loss = np.array(step_g_loss).mean()
        epoch_d_loss = np.array(step_d_loss).mean()
        self.writer.add_scalar('val/epoch_g_loss', epoch_g_loss, epoch)
        self.writer.add_scalar('val/epoch_d_loss', epoch_d_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/f1_score', f1_score, epoch)
            # print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        self.write_log_to_txt(f"Epoch {epoch:3d} {'val'.ljust(6, ' ')} | epoch_g_loss: {epoch_g_loss:4f} epoch_d_loss: {epoch_d_loss:4f}, mIoU: {mIoU:4f}, f1_score: {f1_score:4f}")
        print(f"Epoch {epoch} {'val'.ljust(6, ' ')} | epoch_g_loss: {epoch_g_loss:4f} epoch_d_loss: {epoch_d_loss:4f}, mIoU: {mIoU:4f}, f1_score: {f1_score:4f}")
        
        self.g_valEpoch_loss.append(epoch_g_loss)
        self.d_valEpoch_loss.append(epoch_d_loss)
        # save checkpoint every epoch

        if f1_score > self.best_pred:
            self.best_pred = f1_score
            if f1_score > 0.3:
                is_best_epoch = True
        else:
            is_best_epoch = False

        self.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': self.generator.state_dict(),
            'optimizer': self.optimizer_G.state_dict(),
            'trainingEpoch_loss': self.g_trainingEpoch_loss,
            'valEpoch_loss':self.g_valEpoch_loss,
            'best_pred': self.best_pred,
        }, is_best_epoch, 'generator')

        self.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': self.discriminator.state_dict(),
            'optimizer': self.optimizer_D.state_dict(),
            'trainingEpoch_loss': self.d_trainingEpoch_loss,
            'valEpoch_loss':self.d_valEpoch_loss,
            'best_pred': self.best_pred,
        }, is_best_epoch, 'discriminator')


    def forward(self):
        try:
            for epoch in range(self.start_epoch, self.args.n_epochs):
                self.training(epoch)
                self.validation(epoch)
            plot_loss(self.g_trainingEpoch_loss, self.g_valEpoch_loss, self.exp_dir, 'generator')
            plot_loss(self.d_trainingEpoch_loss, self.d_valEpoch_loss, self.exp_dir, 'discriminator')
        except KeyboardInterrupt:
            plot_loss(self.g_trainingEpoch_loss, self.g_valEpoch_loss, self.exp_dir, 'generator')
            plot_loss(self.d_trainingEpoch_loss, self.d_valEpoch_loss, self.exp_dir, 'discriminator')
        self.write_log_to_txt(f"best f1_score: {self.best_pred:4f}")



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

        self.model_name = 'LikeUNet'
        self.threshold = 0.5
        log_dir = sorted(glob.glob(os.path.join('logs', 'LikeUNet', args.dataset_name, 'run_*')), key=lambda x: int(x.split('_')[-1]))
        run_id = int(log_dir[-1].split('_')[-1]) + 1 if log_dir else 0
        exp_dir = os.path.join('logs', 'LikeUNet', args.dataset_name, 'run_{}'.format(str(run_id)))
        print('exp_dir', exp_dir)
        os.makedirs(exp_dir, exist_ok=True)

        self.logfile = os.path.join(exp_dir, 'log.txt')
        p=vars(self.args)
        log_file = open(self.logfile, "a+")
        for key, val in p.items():
            log_file.write(key + ':' + str(val) + '\n')
        log_file.write('\n')

        output_img_dir = os.path.join(exp_dir, 'result')
        os.makedirs(output_img_dir, exist_ok=True)
        
        self.log_dir = log_dir
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


        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, betas=(args.b1, args.b2))


        self.criterion = SegmentationLosses(cuda=self.cuda, args = args).build_loss(type=args.loss_type)
        
        trainset = CustomDataset(args, split="train")
        testset = CustomDataset(args, split="test")
        # print('trainset', trainset)
        self.trainloader = DataLoader(trainset, batch_size=args.batch_size, drop_last=True, 
                                                shuffle=True, num_workers=2)
        self.testloader = DataLoader(testset, batch_size=args.batch_size, drop_last=True,
                                                shuffle=False, num_workers=2)
        
        self.writer = SummaryWriter(exp_dir)


        self.evaluator = EvaluatorForeground(2)

        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                                args.n_epochs, len(self.trainloader), args = args)
        
        self.best_pred = 0
        self.start_epoch = 0
        self.trainingEpoch_loss = []
        self.valEpoch_loss = []
        self.epoch_f1_score = []
        self.epoch_mIou = []
        g_weight_filepath = args.resume
        fine_tune = False
        if g_weight_filepath and os.path.exists(g_weight_filepath):
            print(f"resume from {g_weight_filepath}")
            checkpoint = torch.load(g_weight_filepath, map_location=torch.device('cpu'))
            self.model.load_state_dict(checkpoint['state_dict'])
            if not fine_tune:
                self.start_epoch = checkpoint['epoch']
                self.best_pred = checkpoint['best_pred']
                self.model.load_state_dict(checkpoint['optimizer'])
                self.trainingEpoch_loss += checkpoint['trainingEpoch_loss']
                self.valEpoch_loss += checkpoint['valEpoch_loss']
                self.epoch_f1_score += checkpoint['epoch_f1_score']
                self.epoch_mIou += checkpoint['epoch_mIou']
                
        

        print(f"train dataset num: {len(self.trainloader)}, test dataset num: {len(self.testloader)}")
        print(f"Starting epoch {self.start_epoch}:\n")

        
                
    def save_checkpoint(self, state, is_best, model_name=''):
        """Saves checkpoint to disk"""
        best_model_filepath = os.path.join(self.exp_dir, f'{model_name}_model_best.pth.tar')
        filename = os.path.join(self.exp_dir, f'{model_name}_checkpoint.pth.tar')
        torch.save(state, filename)
        if is_best:   
            torch.save(state, best_model_filepath)

    def write_log_to_txt(self, data):
        assert isinstance(data, str)
        log_file = open(self.logfile, "a+")
        if data.endswith('\n'):
            log_file.write(data)
        else:
            log_file.write(data+'\n')
        log_file.close()# 

    def training(self, epoch):

        self.model.train()
        tbar = tqdm(self.trainloader)
        # if self.args.master:
        #     print(f'rank {self.args.rank} num_img_tr: {num_img_tr}')
        Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor

        step_loss = []

        # for iter, sample in enumerate(tbar):
        for iter, sample in enumerate(tbar):
            # print(f'rank {self.args.rank} dataload time {round(time.time() - start, 3)}')
            # start = time.time()
            if self.args.with_background:
                image, label, background, img_name = sample['image'], sample['label'], sample['background'], sample['img_name']
                image = torch.cat((image, background), 1)
            else:
                image, label, img_name = sample['image'], sample['label'], sample['img_name']
            image, label = image.to(self.device), label.to(self.device)
            

            self.optimizer.zero_grad()

            output = self.model(image)

            # print('image.shape', image.shape) #torch.Size([16, 6, 256, 256])
            # print('gen_output.shape', gen_output.shape) #torch.Size([16, 1, 256, 256])

            # Loss measures generator's ability to fool the discriminator
            loss = self.criterion(output, label)
            loss.backward()
            self.optimizer.step()
            self.scheduler(self.optimizer, iter, epoch)

            tbar.set_description(f"Epoch {epoch:3d} {'train'.ljust(6, ' ')} | iter {iter:04d} loss: {loss.item():4f}")
            

            step_loss.append(loss.item())

            if self.args.debug and iter > 30:
                break

        epoch_loss = np.array(step_loss).mean()
        self.writer.add_scalar('train/epoch_loss', epoch_loss, epoch)
        self.write_log_to_txt(f"Epoch {epoch:3d} {'train'.ljust(6, ' ')} | epoch_loss: {epoch_loss:4f}")
        # print(f"Epoch {epoch} {'train'.ljust(6, ' ')} | epoch_g_loss: {epoch_g_loss:4f} epoch_d_loss: {epoch_d_loss:4f}")
            # print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        self.trainingEpoch_loss.append(epoch_loss)



    def validation(self, epoch = 0):
        

        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.testloader)
        # if self.args.master:
        #     print(f'rank {self.args.rank} num_img_tr: {num_img_tr}')
        is_best_epoch = False


        step_loss = []

        for iter, sample in enumerate(tbar):
            # print(f'rank {self.args.rank} dataload time {round(time.time() - start, 3)}')
            # start = time.time()
            
            # print('target', target.size(), image.size())
            if self.args.with_background:
                image, label, background, img_name = sample['image'], sample['label'], sample['background'], sample['img_name']
                image = torch.cat((image, background), 1)
            else:
                image, label, img_name = sample['image'], sample['label'], sample['img_name']
            image, label = image.to(self.device), label.to(self.device)


            self.optimizer.zero_grad()
        
            # Loss measures generator's ability to fool the discriminator
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, label)
            pred = output.data.cpu().numpy()
            target = label.cpu().numpy()
            pred = softmax(pred, axis=1)
            pred = np.argmax(pred, axis=1)
            self.evaluator.add_batch(target.astype(int), pred.astype(int))


            

            tbar.set_description(f"Epoch {epoch:3d} {'val'.ljust(6, ' ')} | iter {iter:04d} loss: {loss.item():4f}")

            step_loss.append(loss.item())

            if self.args.debug and iter > 30:
                break

        mIoU = self.evaluator.Mean_Intersection_over_Union()
        f1_score = self.evaluator.F1_Score()
        epoch_loss = np.array(step_loss).mean()
        self.writer.add_scalar('val/epoch_loss', epoch_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/f1_score', f1_score, epoch)
            # print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        self.write_log_to_txt(f"Epoch {epoch:3d} {'val'.ljust(6, ' ')} | epoch_loss: {epoch_loss:4f}, mIoU: {mIoU:4f}, f1_score: {f1_score:4f}")
        print(f"Epoch {epoch} {'val'.ljust(6, ' ')} | epoch_loss: {epoch_loss:4f}, mIoU: {mIoU:4f}, f1_score: {f1_score:4f}")
        
        self.valEpoch_loss.append(epoch_loss)
        self.epoch_mIou.append(mIoU)
        self.epoch_f1_score.append(f1_score)
        # save checkpoint every epoch

        if f1_score > self.best_pred:
            self.best_pred = f1_score
            if f1_score > 0.3:
                is_best_epoch = True
        else:
            is_best_epoch = False

        self.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'trainingEpoch_loss': self.trainingEpoch_loss,
            'valEpoch_loss':self.valEpoch_loss,
            'epoch_mIoU':self.epoch_mIou,
            'epoch_f1_score':self.epoch_f1_score,
            'best_pred': self.best_pred,
        }, is_best_epoch, self.model_name)



    def forward(self):
        try:
            for epoch in range(self.start_epoch, self.args.n_epochs):
                self.training(epoch)
                self.validation(epoch)
            plot_loss(self.trainingEpoch_loss, self.valEpoch_loss, self.exp_dir, self.model_name)
        except KeyboardInterrupt:
            plot_loss(self.trainingEpoch_loss, self.valEpoch_loss, self.exp_dir, self.model_name)
        self.write_log_to_txt(f"best f1_score: {self.best_pred:4f}")



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

        self.model_name = 'VAEUNet'
        self.threshold = 0.5
        log_dir = sorted(glob.glob(os.path.join('logs', self.model_name, args.dataset_name, 'run_*')), key=lambda x: int(x.split('_')[-1]))
        run_id = int(log_dir[-1].split('_')[-1]) + 1 if log_dir else 0
        exp_dir = os.path.join('logs', self.model_name, args.dataset_name, 'run_{}'.format(str(run_id)))
        print('exp_dir', exp_dir)
        os.makedirs(exp_dir, exist_ok=True)

        self.logfile = os.path.join(exp_dir, 'log.txt')
        p=vars(self.args)
        log_file = open(self.logfile, "a+")
        for key, val in p.items():
            log_file.write(key + ':' + str(val) + '\n')
        log_file.write('\n')

        output_img_dir = os.path.join(exp_dir, 'result')
        os.makedirs(output_img_dir, exist_ok=True)
        
        self.log_dir = log_dir
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


        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, betas=(args.b1, args.b2))


        self.seg_criterion = SegmentationLosses(cuda=self.cuda, args = args).build_loss(type=args.loss_type)
        self.kl_criterion = SegmentationLosses(cuda=self.cuda, args = args).build_loss(type='kl')
        self.recons_criterion = SegmentationLosses(cuda=self.cuda, args = args).build_loss(type='mse')

        trainset = CustomDataset(args, split="train")
        testset = CustomDataset(args, split="test")
        # print('trainset', trainset)
        self.trainloader = DataLoader(trainset, batch_size=args.batch_size, drop_last=True, 
                                                shuffle=True, num_workers=2)
        self.testloader = DataLoader(testset, batch_size=args.batch_size, drop_last=True,
                                                shuffle=False, num_workers=2)
        
        self.writer = SummaryWriter(exp_dir)


        self.evaluator = EvaluatorForeground(2)

        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                                args.n_epochs, len(self.trainloader), args = args)
        
        self.best_pred = 0
        self.start_epoch = 0
        self.trainingEpoch_loss = []
        self.trainingEpoch_seg_loss = []
        self.trainingEpoch_rec_loss = []
        self.trainingEpoch_kl_loss = []

        self.valEpoch_loss = []
        self.valEpoch_seg_loss = []
        self.valEpoch_rec_loss = []
        self.valEpoch_kl_loss = []

        self.epoch_f1_score = []
        self.epoch_mIou = []
        g_weight_filepath = args.resume
        fine_tune = False
        if g_weight_filepath and os.path.exists(g_weight_filepath):
            print(f"resume from {g_weight_filepath}")
            checkpoint = torch.load(g_weight_filepath, map_location=torch.device('cpu'))
            self.model.load_state_dict(checkpoint['state_dict'])
            if not fine_tune:
                self.start_epoch = checkpoint['epoch']
                self.best_pred = checkpoint['best_pred']
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                # self.trainingEpoch_loss += checkpoint['trainingEpoch_loss']
                # self.valEpoch_loss += checkpoint['valEpoch_loss']
                # self.epoch_f1_score += checkpoint['epoch_f1_score']
                # self.epoch_mIou += checkpoint['epoch_mIou']
                
        

        print(f"train dataset num: {len(self.trainloader)}, test dataset num: {len(self.testloader)}")
        print(f"Starting epoch {self.start_epoch}:\n")

        
                
    def save_checkpoint(self, state, is_best, model_name=''):
        """Saves checkpoint to disk"""
        best_model_filepath = os.path.join(self.exp_dir, f'{model_name}_model_best.pth.tar')
        filename = os.path.join(self.exp_dir, f'{model_name}_checkpoint.pth.tar')
        torch.save(state, filename)
        if is_best:   
            torch.save(state, best_model_filepath)

    def write_log_to_txt(self, data):
        assert isinstance(data, str)
        log_file = open(self.logfile, "a+")
        if data.endswith('\n'):
            log_file.write(data)
        else:
            log_file.write(data+'\n')
        log_file.close()# 

    def training(self, epoch):

        self.model.train()
        tbar = tqdm(self.trainloader)
        # if self.args.master:
        #     print(f'rank {self.args.rank} num_img_tr: {num_img_tr}')


        step_loss = []
        seg_step_loss = []
        rec_step_loss = []
        kl_step_loss = []

        epoch_cnt = epoch - self.start_epoch
        lambda_seg = 1000 - 50*epoch_cnt if 1000 - 50*epoch_cnt > 500 else 500
        lambda_kl = 0.01*(10*epoch_cnt + 1) if 0.01*(10*epoch_cnt + 1) <= 10 else 10
        lambda_rec = 0.01

        # for iter, sample in enumerate(tbar):
        for iter, sample in enumerate(tbar):
            # print(f'rank {self.args.rank} dataload time {round(time.time() - start, 3)}')
            # start = time.time()
            if self.args.with_background:
                image_f, label, background, img_name = sample['image'], sample['label'], sample['background'], sample['img_name']
                image = torch.cat((image_f, background), 1)
                image = image.to(self.device)
                image_f = image_f.to(self.device)
            else:
                image_f, label, img_name = sample['image'], sample['label'], sample['img_name']
                image_f = image_f.to(self.device)
                image = image_f
            
            label = label.to(self.device)

            self.optimizer.zero_grad()

            seg, p_x, mu, logvar = self.model(image)

            # print('image.shape', image.shape) #torch.Size([16, 6, 256, 256])
            # print('gen_output.shape', gen_output.shape) #torch.Size([16, 1, 256, 256])
            # print('label.shape', label.shape) #([32, 256, 256])
            # print('p_x.shape', p_x.shape) #torch.Size([32, 3, 256, 256])
            # print('mu.shape', mu.shape) #torch.Size([32, 256])
            # print('logvar.shape', logvar.shape) #torch.Size([32, 256])

            # Loss measures generator's ability to fool the discriminator
            seg_loss = self.seg_criterion(seg, label)
            
            if self.args.reconstract_whole_image:
                rec_loss = self.recons_criterion(p_x, image_f)
            else:
                temp_lable = torch.unsqueeze(label, 1).repeat(1,3,1,1)
                rec_loss = self.recons_criterion(p_x.mul(temp_lable), image_f.mul(temp_lable))
            kl_loss = self.kl_criterion(mu, logvar) 
            # print('seg_loss', seg_loss) # tensor(0.0625, device='cuda:0', grad_fn=<MulBackward0>)
            # print('rec_loss', rec_loss.shape) # tensor(207.9003, device='cuda:0', grad_fn=<MeanBackward1>)
            # print('kl_loss', kl_loss) #
            loss = seg_loss*lambda_seg + rec_loss*lambda_rec + kl_loss*lambda_kl
            loss.backward()
            self.optimizer.step()
            self.scheduler(self.optimizer, iter, epoch)

            tbar.set_description(f"Epoch {epoch:3d} {'train'.ljust(6, ' ')} | iter {iter:04d} seg_loss: {seg_loss.item():4f}, rec_loss: {rec_loss.item():4f}, kl_loss: {kl_loss.item():4f}, loss: {loss.item():4f}")
            

            step_loss.append(loss.item())
            seg_step_loss.append(seg_loss.item())
            rec_step_loss.append(rec_loss.item())
            kl_step_loss.append(kl_loss.item())


            if self.args.debug and iter > 30:
                break

        epoch_loss = np.array(step_loss).mean()
        epoch_seg_loss = np.array(seg_step_loss).mean()
        epoch_rec_loss = np.array(rec_step_loss).mean()
        epoch_kl_loss = np.array(kl_step_loss).mean()

        self.writer.add_scalar('train/epoch_loss', epoch_loss, epoch)
        self.writer.add_scalar('train/epoch_seg_loss', epoch_seg_loss, epoch)
        self.writer.add_scalar('train/epoch_rec_loss', epoch_rec_loss, epoch)
        self.writer.add_scalar('train/epoch_kl_loss', epoch_kl_loss, epoch)

        self.write_log_to_txt(f"Epoch {epoch:3d} {'train'.ljust(6, ' ')} | epoch_loss: {epoch_loss:4f}")
        self.write_log_to_txt(f"Epoch {epoch:3d} {'train'.ljust(6, ' ')} | epoch_seg_loss: {epoch_seg_loss:4f}")
        self.write_log_to_txt(f"Epoch {epoch:3d} {'train'.ljust(6, ' ')} | epoch_rec_loss: {epoch_rec_loss:4f}")
        self.write_log_to_txt(f"Epoch {epoch:3d} {'train'.ljust(6, ' ')} | epoch_kl_loss: {epoch_kl_loss:4f}")
        # print(f"Epoch {epoch} {'train'.ljust(6, ' ')} | epoch_g_loss: {epoch_g_loss:4f} epoch_d_loss: {epoch_d_loss:4f}")
            # print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        self.trainingEpoch_loss.append(epoch_loss)
        self.trainingEpoch_seg_loss.append(epoch_seg_loss)
        self.trainingEpoch_rec_loss.append(epoch_rec_loss)
        self.trainingEpoch_kl_loss.append(epoch_kl_loss)



    def validation(self, epoch = 0):
        

        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.testloader)
        # if self.args.master:
        #     print(f'rank {self.args.rank} num_img_tr: {num_img_tr}')
        is_best_epoch = False


        step_loss = []
        seg_step_loss = []
        rec_step_loss = []
        kl_step_loss = []
        epoch_cnt = epoch - self.start_epoch

        lambda_seg = 1000 - 50*epoch_cnt if 1000 - 50*epoch_cnt > 500 else 500
        lambda_kl = 0.01*(10*epoch_cnt + 1) if 0.01*(10*epoch_cnt + 1) <= 10 else 10
        lambda_rec = 0.01

        for iter, sample in enumerate(tbar):
            # print(f'rank {self.args.rank} dataload time {round(time.time() - start, 3)}')
            # start = time.time()
            
            # print('target', target.size(), image.size())
            if self.args.with_background:
                image_f, label, background, img_name = sample['image'], sample['label'], sample['background'], sample['img_name']
                image = torch.cat((image_f, background), 1)
                image = image.to(self.device)
                image_f = image_f.to(self.device)
            else:
                image_f, label, img_name = sample['image'], sample['label'], sample['img_name']
                image_f = image_f.to(self.device)
                image = image_f
            
            label = label.to(self.device)


            self.optimizer.zero_grad()
        
            # Loss measures generator's ability to fool the discriminator
            with torch.no_grad():
                seg, p_x, mu, logvar = self.model(image)

            seg_loss = self.seg_criterion(seg, label)
            if self.args.reconstract_whole_image:
                rec_loss = self.recons_criterion(p_x, image_f)
            else:
                temp_lable = torch.unsqueeze(label, 1).repeat(1,3,1,1)
                rec_loss = self.recons_criterion(p_x.mul(temp_lable), image_f.mul(temp_lable))
                
            kl_loss = self.kl_criterion(mu, logvar)
            loss = seg_loss*lambda_seg + rec_loss*lambda_rec + kl_loss*lambda_kl


            pred = seg.data.cpu().numpy()
            target = label.cpu().numpy()
            pred = softmax(pred, axis=1)
            pred = np.argmax(pred, axis=1)
            self.evaluator.add_batch(target.astype(int), pred.astype(int))


            tbar.set_description(f"Epoch {epoch:3d} {'val'.ljust(6, ' ')} | iter {iter:04d} seg_loss: {seg_loss.item():4f}, rec_loss: {rec_loss.item():4f}, kl_loss: {kl_loss.item():4f}, loss: {loss.item():4f}")

            step_loss.append(loss.item())
            seg_step_loss.append(seg_loss.item())
            rec_step_loss.append(rec_loss.item())
            kl_step_loss.append(kl_loss.item())

            if self.args.debug and iter > 30:
                break

        mIoU = self.evaluator.Mean_Intersection_over_Union()
        f1_score = self.evaluator.F1_Score()
        epoch_loss = np.array(step_loss).mean()
        epoch_seg_loss = np.array(seg_step_loss).mean()
        epoch_rec_loss = np.array(rec_step_loss).mean()
        epoch_kl_loss = np.array(kl_step_loss).mean()

        self.writer.add_scalar('val/epoch_loss', epoch_loss, epoch)
        self.writer.add_scalar('val/epoch_seg_loss', epoch_seg_loss, epoch)
        self.writer.add_scalar('val/epoch_rec_loss', epoch_rec_loss, epoch)
        self.writer.add_scalar('val/epoch_kl_loss', epoch_kl_loss, epoch)

        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/f1_score', f1_score, epoch)

            # print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        self.write_log_to_txt(f"Epoch {epoch:3d} {'val'.ljust(6, ' ')} | epoch_loss: {epoch_loss:4f}, mIoU: {mIoU:4f}, f1_score: {f1_score:4f}")
        self.write_log_to_txt(f"Epoch {epoch:3d} {'val'.ljust(6, ' ')} | epoch_seg_loss: {epoch_seg_loss:4f}")
        self.write_log_to_txt(f"Epoch {epoch:3d} {'val'.ljust(6, ' ')} | epoch_rec_loss: {epoch_rec_loss:4f}")
        self.write_log_to_txt(f"Epoch {epoch:3d} {'val'.ljust(6, ' ')} | epoch_kl_loss: {epoch_kl_loss:4f}")
        print(f"Epoch {epoch} {'val'.ljust(6, ' ')} | epoch_loss: {epoch_loss:4f}, mIoU: {mIoU:4f}, f1_score: {f1_score:4f}")
        
        self.valEpoch_loss.append(epoch_loss)
        self.valEpoch_seg_loss.append(epoch_seg_loss)
        self.valEpoch_rec_loss.append(epoch_rec_loss)
        self.valEpoch_kl_loss.append(epoch_kl_loss)

        self.epoch_mIou.append(mIoU)
        self.epoch_f1_score.append(f1_score)
        # save checkpoint every epoch

        if f1_score > self.best_pred:
            self.best_pred = f1_score
            if f1_score > 0.3:
                is_best_epoch = True
        else:
            is_best_epoch = False

        self.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'trainingEpoch_loss': self.trainingEpoch_loss,
            'valEpoch_loss':self.valEpoch_loss,
            'epoch_mIoU':self.epoch_mIou,
            'epoch_f1_score':self.epoch_f1_score,
            'best_pred': self.best_pred,
        }, is_best_epoch, self.model_name)

    
       
    def forward(self):
        try:
            cnt = 1
            for epoch in range(self.start_epoch, self.args.n_epochs):
                self.training(epoch)
                self.validation(epoch)
                cnt += 1
                if self.args.debug and cnt > 3:
                    break
            plot_loss(self.trainingEpoch_loss, self.valEpoch_loss, self.exp_dir, f'{self.model_name}', 'total')
            plot_loss(self.trainingEpoch_seg_loss, self.valEpoch_seg_loss, self.exp_dir, f'{self.model_name}', 'seg')
            plot_loss(self.trainingEpoch_rec_loss, self.valEpoch_rec_loss, self.exp_dir, f'{self.model_name}', 'recons')
            plot_loss(self.trainingEpoch_kl_loss, self.valEpoch_kl_loss, self.exp_dir, f'{self.model_name}', 'kl')
            plot_mIou_f1score(self.epoch_mIou, self.epoch_f1_score, self.exp_dir, self.model_name)
        except KeyboardInterrupt:
            plot_loss(self.trainingEpoch_loss, self.valEpoch_loss, self.exp_dir, f'{self.model_name}', 'total')
            plot_loss(self.trainingEpoch_seg_loss, self.valEpoch_seg_loss, self.exp_dir, f'{self.model_name}', 'seg')
            plot_loss(self.trainingEpoch_rec_loss, self.valEpoch_rec_loss, self.exp_dir, f'{self.model_name}', 'recons')
            plot_loss(self.trainingEpoch_kl_loss, self.valEpoch_kl_loss, self.exp_dir, f'{self.model_name}', 'kl')
            plot_mIou_f1score(self.epoch_mIou, self.epoch_f1_score, self.exp_dir, self.model_name)
        self.write_log_to_txt(f"best f1_score: {self.best_pred:4f}")

            
def main(args):
    if args.model_name == 'bscGAN':
        worker = bscGanWorker(args)
        worker.forward()
    elif args.model_name == 'LikeUNet':
        worker = LikeUNetWorker(args)
        worker.forward()
    elif args.model_name == 'VAEUNet':
        worker = VAEUNetWorker(args)
        worker.forward()

if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=55, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0005, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument('--lr_scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')

    parser.add_argument("--img_size", type=int, default=256, help="size of each image dimension")
    parser.add_argument("--ignore_index", type=int, default=255, help="ignore_index")
    parser.add_argument('--with_background', action='store_true', default=False, #True, False
                    help='with_background image')
    
    parser.add_argument('--resume', type=str, default=None, help='resume')
    parser.add_argument('--resume2', type=str, default=None, help='resume')
    parser.add_argument('--ft', action='store_true', default=False,
                    help='Fine tune')
    parser.add_argument('--debug', action='store_true', default=False, #True, False
                    help='debug mode')
    parser.add_argument('--reconstract_whole_image', action='store_true', default=False, #True, False
                    help='reconstract_whole_image')
    # parser.add_argument('--model_name', type=str, default='bscGAN', help='model_name')
    # parser.add_argument('--model_name', type=str, default='LikeUNet', help='model_name')
    parser.add_argument('--model_name', type=str, default='VAEUNet', help='model_name, ie. bscGAN, LikeUNet, or VAEUNet')
    parser.add_argument('--loss_type', type=str, default='focal', help='loss_type')
    parser.add_argument('--dataset_name', type=str, default='bmc', help='cdnet or bmc')
    parser.add_argument('--dataset_dir', type=str, default='data/bmc', help='dataset dir, data/CDnet2014 or data/bmc')
    
    opt = parser.parse_args()

    main(opt)
    