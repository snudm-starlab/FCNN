# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

import os
from re import I
import sys
import argparse
import time
from datetime import datetime
from tqdm import tqdm
import copy

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.fourier2 import *
from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights

@torch.no_grad()
def _eval():
    tot_mse = 0; count = 0
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):
        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()

        inputs = prev_layers(images)
        _target = target_layer(inputs)
        _out = fourier_layer(inputs)
        _diff = _target - _out
        _mse = (_diff * _diff).mean()
        tot_mse += _mse
        count += 1
    return tot_mse / count

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='resnet34', help='net type')
    parser.add_argument('-seed', type=int, default=0, help='random seed for reproducibility')
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-lr', type=float, default=5e-6, help='initial learning rate')
    parser.add_argument('-epoch', type=int, default=30, help='epochs to train')
    parser.add_argument('-load', action='store_true', default=True, help='load pretrained or not')
    args = parser.parse_args()

    ###################################
    print("="*80)
    print(args)
    print("="*80)

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)
    ###################################

    net = get_network(args)
    ############### Load weights and get target layers  ############################
    if args.load:
        weight_path = 'checkpoint/resnet34/test.pth'
        net.load_state_dict(torch.load(weight_path)) 
    # print(net)
    ############################################################

    #data preprocessing:
    cifar100_training_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )
    ############################# Prepare training #####################################

    prev_layers = nn.Sequential(
            net.conv1,
            )
    target_layer = nn.Sequential(
            net.conv2_x[0].residual_function[0], # Conv2d
            # net.conv2_x[0].residual_function[1], # BatchNorm
            # net.conv2_x[0].residual_function[2], # ReLU
            )
    fourier_layer = nn.Sequential(
            FConv2d(32, 64, 1, 1),
            # copy.deepcopy(net.conv2_x[0].residual_function[1]), # BatchNorm
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            FConv2d(32, 64, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            FConv2d(32, 64, 1, 1),
            # nn.BatchNorm2d(64),
            )
    print("* Target layer: ", target_layer)
    print("* Fourier layer: ", fourier_layer)

    # print(target_layer)
    # fourier_layer = FConv2d(32, 64, 1, 1)

    prev_layers.eval()
    target_layer.eval()
    fourier_layer.train()
    prev_layers.cuda()
    target_layer.cuda()
    fourier_layer.cuda()

    # optimizer = optim.SGD(fourier_layer.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # optimizer = optim.SGD([fourier_layer.weight, target_layer.weight], lr=args.lr)
    optimizer = optim.SGD(fourier_layer.parameters(), lr=args.lr)
    # test_input = torch.randn(1, 3,32,32)
    # _feature = prev_layers(test_input)
    # target_feature = target_layer(_feature)
    # _feature = fourier_layer(_feature)
    # print("target feature: ", target_feature.shape)
    # print("generated feature: ", _feature.shape) 
    #####################################################################################
    
    for epoch in range(1, args.epoch + 1):
        # print(f"Epoch #{epoch:2d} |", end = '')
        train_mse = 0; _count = 0; orig_output = 0; fourier_output = 0
        for batch_index, (images, labels) in tqdm(enumerate(cifar100_training_loader)):
            if args.gpu:
                labels = labels.cuda()
                images = images.cuda()
            inputs = prev_layers(images).detach()
            _target = target_layer(inputs).detach()
            _out = fourier_layer(inputs)
            # print("_target: ", (_target*_target).mean())
            _diff = _target - _out
            loss = (_diff * _diff).mean()
            # loss = torch.sqrt((_diff * _diff).mean())
            train_mse += loss
            _count += 1
            orig_output += (_target * _target).mean().detach()
            fourier_output += (_out * _out).mean().detach()
            # print(f"* MSE: {loss:.4f}")
            loss.backward()
            optimizer.step()
        test_mse = _eval()
        print(f"{epoch:3d} | orig_output: {orig_output/_count:.5f} | fourier_output: {fourier_output/_count:.5f}",
              f"| train_mse: {train_mse/_count:.5f} | test_mse = {test_mse:.5f})")

    # best_acc = 0.0
    """
    for epoch in range(1, args.epoch + 1):
        pass
        # train(epoch)
        # acc = eval_training(epoch, thres=args.thres)

        #start to save best performance model after learning rate decay to 0.01
    """



