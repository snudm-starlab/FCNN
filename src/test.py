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
# test.py
# - codes for testing trained models
#
# This code is mainly based on the [GitHub Repository]
# [GitHub Repository]: https://github.com/weiaicunzai/pytorch-cifar100
################################################################################

import argparse
from datetime import datetime
from matplotlib import pyplot as plt

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utils import \
        get_network, get_test_dataloader, get_flops
import numpy as np
from tqdm import tqdm

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

if __name__ == '__main__':
    """ main function for test """    
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-weights', type=str, required=True, 
                        help='the weights file you want to test')
    # arguments for FCNN
    parser.add_argument('-nu', type=int, default=-1, 
                        help='Coefficient for choosing the number of filters')
    parser.add_argument('-rho', type=int, default=-1, 
                        help='Coefficient for choosing kenel size')
    args = parser.parse_args()

    # generate networks
    net = get_network(args)
    
    # get test data loader
    cifar100_test_loader = get_test_dataloader(
        Settings.CIFAR100_TRAIN_MEAN,
        Settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
    )
    
    # load a trained model to evaluate
    net.load_state_dict(torch.load(args.weights))
    net.eval()
    
    # evaluate 
    correct = 0.0; total = 0
    with torch.no_grad():
        for n_iter, (image, label) in enumerate(cifar100_test_loader):
            if args.gpu:
                image = image.cuda()
                label = label.cuda()
            output = net(image)
            pred = output.argmax(dim=1)
            correct += pred.eq(label).float().sum()
    _acc = correct / len(cifar100_test_loader.dataset) * 100

    # print results
    print()
    orig_acc = 77.77; orig_params = 21.33; orig_flops=2.32043
    _params = np.sum([_p.numel() for _n, _p in net.named_parameters()])/1e6
    _flops = get_flops(net, args, imagenet=False)
    print("="*30)     
    print("* After compression w/ FCNN")
    print(f"  + Accuracy: {_acc:.2f} ({_acc-orig_acc:.2f})")
    print(f"  + Parameters: {_params:.2f}M ({orig_params/_params:.2f}x)")
    print(f"  + FLOPs: {_flops:.2f}G ({orig_flops/_flops:.2f}x)")
    print("="*30)
