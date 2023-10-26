#############################################################################
# Starlab CNN Compression with FCNN (Flexible Convolutional Neural Network)
# Author: Seungcheol Park (ant6si@snu.ac.kr), Seoul National University
#         U Kang (ukang@snu.ac.kr), Seoul National University
# Version : 1.0
# Date : Oct 26, 2023
# Main Contact: Seungcheol Park
# This software is free of charge under research purposes.
# For commercial purposes, please contact the authors.
# 
# utils.py
# - utilization functions to help train and test
#
# This code is mainly based on the [GitHub Repository]
# [GitHub Repository]: https://github.com/weiaicunzai/pytorch-cifar100
################################################################################

import os
import sys
import re
from datetime import datetime

from torch.autograd import Variable
import numpy

import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_network(args):
    """
    return an arhictecture of a given neural network
    * Inputs:
        - args: set of arguments including the desired network type
    * Outputs:
       - net: a neural network 
    """
    if args.net == 'resnet34':
        from models.resnet import resnet34
        net = resnet34()
    elif args.net == 'fcnn34':
        from models.fcnn import fcnn34
        net = fcnn34(nu=args.nu, rho=args.rho)
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()
    if args.gpu:
        net = net.cuda()

    return net


def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """
    return a dataloader for training
    * Inputs:
        - mean: preset mean of cifar100 training dataset
        - std: preset std of cifar100 training dataset
        - path: path to cifar100 training python dataset
        - batch_size: dataloader batchsize
        - num_workers: dataloader num_works
        - shuffle: whether to shuffle
    * Outputs:
       - train_data_loader: a dataloader for training 
    """

    # transformation for images
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    cifar100_training = torchvision.datasets.CIFAR100(root='../data', train=True, 
                                                      download=True, transform=transform_train)
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, 
        num_workers=num_workers, batch_size=batch_size)
    return cifar100_training_loader

def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """
    return a dataloader for testing
    * Inputs:
        - mean: preset mean of cifar100 training dataset
        - std: preset std of cifar100 training dataset
        - path: path to cifar100 testingg python dataset
        - batch_size: dataloader batchsize
        - num_workers: dataloader num_works
        - shuffle: whether to shuffle
    * Outputs:
       - train_data_loader: a dataloader for testing 
    """
    # transformation for images
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    cifar100_test = torchvision.datasets.CIFAR100(root='../data', train=False,
                                                  download=True, transform=transform_test)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle,
        num_workers=num_workers, batch_size=batch_size)
    return cifar100_test_loader



def get_flops(net, args, imagenet=False):
    """
    return FLOPs of a given neural network
    * Inputs:
        - net: a neural network to estimate FLOPs
        - args: a set of arguments
    * Outputs:
       - total_flops: FLOPs of net
    """

    multiply_adds = True
    list_conv3d = []

    def conv3d_hook(self, _input, output):
        """
        a hook function for calculating FLOPs of 3d convolution for FConv layers
        * Inputs:
            - _input: an input of the module
            - _output: an output of the module
        """
        batch_size, _, _, _, _ = _input[0].size()
        output_channels, output_depth, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2] *\
                     (self.in_channels / self.groups) * (2 if multiply_adds else 1) *\
                     args.nu/args.rho                   
                    
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width * output_depth

        list_conv3d.append(flops)

    list_conv = []

    def conv_hook(self, input, output):
        """
        a hook function for calculating FLOPs of 2d convolution layers
        * Inputs:
            - _input: an input of the module
            - _output: an output of the module
        """
        batch_size, _, _, _ = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] *\
                     (self.in_channels / self.groups) * (2 if multiply_adds else 1)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width

        list_conv.append(flops)

    list_linear = []

    def linear_hook(self, input, output):
        """
        a hook function for calculating FLOPs of linear layers
        * Inputs:
            - _input: an input of the module
            - _output: an output of the module
        """
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()

        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)

    list_bn = []

    def bn_hook(self, input, output):
        """
        a hook function for calculating FLOPs of batch normalization (BN) layers
        * Inputs:
            - _input: an input of the module
            - _output: an output of the module
        """
        list_bn.append(input[0].nelement())

    list_relu = []

    def relu_hook(self, input, output):
        """
        a hook function for calculating FLOPs of ReLU layers
        * Inputs:
            - _input: an input of the module
            - _output: an output of the module
        """
        list_relu.append(input[0].nelement())

    list_pooling = []

    def pooling_hook(self, input, output):
        """
        a hook function for calculating FLOPs of pooling layers
        * Inputs:
            - _input: an input of the module
            - _output: an output of the module
        """
        batch_size, _, _, _ = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width

        list_pooling.append(flops)

    def _register_hooks(net):
        """
        calculate the total number of FLOPs of a net
        * Inputs:
            - net: a neural network to estimate FLOPs
        """
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv3d):
                net.register_forward_hook(conv3d_hook)
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            if isinstance(net, torch.nn.BatchNorm2d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)
            if isinstance(net, (torch.nn.MaxPool2d, torch.nn.AvgPool2d)):
                net.register_forward_hook(pooling_hook)
            return
        for c in childrens:
            _register_hooks(c)
    
    # register hook functions
    _register_hooks(net)
    # dummy imput for estimating FLOPs
    input = Variable(torch.rand(3, 32, 32).unsqueeze(0), requires_grad=True).cuda()
    # compute FLOPs
    net(input)
    total_flops = (sum(list_conv3d) + sum(list_conv) + sum(list_linear)
                   + sum(list_bn) + sum(list_relu) + sum(list_pooling))
    total_flops /= 1e9
    return total_flops
