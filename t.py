#test.py
#!/usr/bin/env python3

""" test neuron network performace
print top1 and top5 err on test dataset
of a model

author baiyu
"""

import argparse
from tqdm import tqdm
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from conf import settings
from utils import get_network, get_test_dataloader
from models.fourier import *
import numpy as np
import time

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    parser.add_argument('-thres', type=float, default=1e-2, help='threshold for pruning')
    args = parser.parse_args()

    net = get_network(args)

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        #settings.CIFAR100_PATH,
        num_workers=4,
        batch_size=args.b,
    )

    t_num = np.sum([_p.numel() for _p in net.parameters()])
    print(f"Orig params: {t_num/1e6:.2f}M")
         
    _bs = [1,3,4,6,3]
    for _l in [2, 3, 4, 5]:
    # for _l in [2]:
        b_list = np.arange(_bs[_l-1])
        for _b in b_list:
            for _n in [0, 3]:
                # Index of convolution in the Sequential Function
                # if _l==4 and _b == 0 and _n ==0:
                #     continue
                net = convert(net, _l, _b, _n)

    net.load_state_dict(torch.load(args.weights))
    thr = args.thres
    _num = 0
    t_num = 0
    with torch.no_grad():
        for _n, _p in net.named_parameters():
            t_num += _p.numel()
            if 'wr' in _n or 'wi' in _n:
                _p[_p.abs()<thr] = 0.
                _num += (_p.abs()>thr).sum()
                # print("_n: ", _n)
            else:
                _num += _p.numel()

    print(f"Num params: {_num/1e6:.2f}M | tot params: {t_num/1e6:.2f}M")
        
    
    # convert(net, 2, 2, 0)
    
    """
    for _n, _p in net.named_parameters():
        print(_n ,_p.shape, _p.numel()/1e6)
    """
    # raise Exception("-------------- Test ---------------------")

    # print(net)
    net.eval()

    if args.gpu:
        net = net.cuda()
    
    correct_1 = 0.0
    correct_5 = 0.0
    total = 0
    st = time.time()
    with torch.no_grad():
        # for n_iter, (image, label) in tqdm(enumerate(cifar100_test_loader)):
        for n_iter, (image, label) in enumerate(cifar100_test_loader):
            # print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(cifar100_test_loader)))

            if args.gpu:
                image = image.cuda()
                label = label.cuda()
                # print('GPU INFO.....')
                # print(torch.cuda.memory_summary(), end='')


            output = net(image)
            _, pred = output.topk(5, 1, largest=True, sorted=True)

            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()

            #compute top 5
            correct_5 += correct[:, :5].sum()

            #compute top1
            correct_1 += correct[:, :1].sum()
    
    
    print("Eval Time: ", time.time()-st)
    print("Top 1 err: ", 1 - correct_1 / len(cifar100_test_loader.dataset))
    print("Top 5 err: ", 1 - correct_5 / len(cifar100_test_loader.dataset)) 
    print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))

    
