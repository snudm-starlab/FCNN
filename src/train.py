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
# train.py
# - codes for training
# - parsing arguments, loading data and model, and logging
#
# This code is mainly based on the [GitHub Repository]
# [GitHub Repository]: https://github.com/weiaicunzai/pytorch-cifar100
################################################################################

from tqdm import tqdm
from datetime import datetime
import os, sys, argparse, time, random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from utils import * 
from models.resnet import resnet34

class Settings:
    """ Settings for preprocessing, training, and logging"""
    CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    CHECKPOINT_PATH = '../checkpoint'
    EPOCH = 200
    MILESTONES = [60, 120, 160]
    DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
    TIME_NOW = datetime.now().strftime(DATE_FORMAT)
    SAVE_EPOCH = 10

def train(epoch, _net, _train_loader, _train_scheduler, _opt, _loss):
    """
    a function for training a given net for an epoch

    * Inputs:
       - epoch: numbering of epochs
       - _net: neural network to train
       - _train_loader: data loader for training
       - train_scheduler: a training scheduler
       - _opt: an optimizer for training
       - _loss: a loss function
    * Outputs:
       - _net: a trained neural network
    """
    _net.train()
    for batch_index, (images, labels) in \
            tqdm(enumerate(_train_loader)):
        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        # forwarding
        outputs = _net(images)
        # compute loss
        loss = _loss(outputs, labels)

        # compute loss for KD if we need
        if args.alpha != -1:
            with torch.no_grad():
                t_outputs = teacher(images)
            kl_div = F.kl_div(
                    F.log_softmax(outputs/args.tau, dim=1),
                    F.softmax(t_outputs/args.tau, dim=1),
                    reduction='batchmean'
                    ) * (args.tau * args.tau)
            loss = args.alpha * kl_div + (1-args.alpha) * loss 
        # backward
        loss.backward()
        # update weight
        optimizer.step() 
    # step scheduler after epoch
    _train_scheduler.step()
    return _net

@torch.no_grad()
def eval_training(epoch, _net, _test_loader, _loss):
    """
    a function for testting a neural network

    * Inputs:
       - epoch: numbering of epochs
       - _net: neural network to train
       - _train_loader: data loader for training
       - _opt: an optimizer for training
       - _loss: a loss function
    * Outputs:
       - _net: a trained neural network
    """

    test_loss = 0.0; correct = 0.0
    _net.eval(); start = time.time()
    for (images, labels) in _test_loader:

        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = _net(images)
        loss = _loss(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    print('Test Result: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'\
        .format(
        epoch,
        test_loss / len(_test_loader.dataset),
        correct.float() / len(_test_loader.dataset),
        finish - start
    ))
    return correct.float() / len(cifar100_test_loader.dataset)

if __name__ == '__main__':
    """ main function for training """    
    # Parsing arguments
    parser = argparse.ArgumentParser()
    # general arguments
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    # arguments for KD
    parser.add_argument('-teacher', type=str, default='../checkpoint/resent34/teacher.pth', 
                        help='a path for teacher ResNet34 model')
    parser.add_argument('-alpha', type=float, default=-1, 
                        help='balance coefficient for knowledge distillation')
    parser.add_argument('-tau', type=float, default=-1, 
                        help='temperature of the softmax for knowledge distillation')
    # arguments for FCNN
    parser.add_argument('-nu', type=int, default=-1, 
                        help='a coefficient for choosing the number of filters')
    parser.add_argument('-rho', type=int, default=-1, 
                        help='a coefficient for choosing kenel size')
    args = parser.parse_args()

    # update dataloader
    net = get_network(args)

    # get data loaders
    cifar100_training_loader = get_training_dataloader(
        Settings.CIFAR100_TRAIN_MEAN,
        Settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )

    cifar100_test_loader = get_test_dataloader(
        Settings.CIFAR100_TRAIN_MEAN,
        Settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )
    
    # Setting for training
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=Settings.MILESTONES, gamma=0.2) 
    checkpoint_path = os.path.join(Settings.CHECKPOINT_PATH, args.net, Settings.TIME_NOW)
    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, 'best_model.pth')

    # load teacher 
    if args.alpha != -1:
        teacher_path = args.teacher
        teacher = resnet34()
        teacher.load_state_dict(torch.load(teacher_path))
        teacher = teacher.cuda()
        teacher.eval()
        print("* Load teacher done!!")
    
    # training
    best_acc = 0.0
    for epoch in range(1, Settings.EPOCH + 1):
        net = train(epoch, net, cifar100_training_loader, train_scheduler, 
                    optimizer, loss_function)
        acc = eval_training(epoch, net, cifar100_test_loader, loss_function)

        # if epoch > Settings.MILESTONES[1] and best_acc < acc:
        if best_acc < acc:
            best_acc = acc
            print('* update best ckpt'.format(checkpoint_path))
            torch.save(net.state_dict(), checkpoint_path)
            continue
    
    # print training results
    num_params = np.sum([_p.numel() for _p in net.parameters()]) / 1e6
    _flops = get_flops(net, args, imagenet=False)
    print("="*50)
    print(args)
    print("*** Best acc: ", best_acc)
    print(f"** # Params: {num_params:.5f}M")
    print(f"** # FLOPs: {_flops:.2f}G")
    print("="*50)
    
