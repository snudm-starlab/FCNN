#test.py
#!/usr/bin/env python3

""" test neuron network performace
print top1 and top5 err on test dataset
of a model

author baiyu
"""

import argparse

from matplotlib import pyplot as plt

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from conf import settings
from utils import get_network, get_test_dataloader
import numpy as np
from tqdm import tqdm
from flops_counter import get_flops

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-rho', type=float, default=6, help='width')
    parser.add_argument('-nu', type=float, default=2, help='sharing')
    # parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    # parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    args = parser.parse_args()
    # args.weights = None
    args.gpu = True
    args.b = 10
    net = get_network(args)
    
    # print("* Num params: ", np.sum([_p.numel() for _n, _p in net.named_parameters()])/1e6)
    # raise Exception


    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        #settings.CIFAR100_PATH,
        num_workers=4,
        batch_size=args.b,
    )

    net.load_state_dict(torch.load(args.weights))
    net.eval()

    correct = 0.0
    total = 0

    with torch.no_grad():
        # for n_iter, (image, label) in tqdm(enumerate(cifar100_test_loader)):
        for n_iter, (image, label) in enumerate(cifar100_test_loader):
            if args.gpu:
                image = image.cuda()
                label = label.cuda()

            output = net(image)
            pred = output.argmax(dim=1)
            # print(f"* ({n_iter}) truth labels:", label)
            # print(f"* ({n_iter}) predicted labels:", pred, "\n")
            correct += pred.eq(label).float().sum()

    _acc = correct / len(cifar100_test_loader.dataset) * 100

    print()
    orig_acc = 77.77; orig_params = 21.33; orig_flops=2.32043

    _params = np.sum([_p.numel() for _n, _p in net.named_parameters()])/1e6
    _flops = get_flops(net, args, imagenet=False)
    print("="*30)    
    """
    print("* Original Model")
    print(f"  + Accuracy: {_acc:.2f}")
    print(f"  + Parameters: {_params:.2f}M")
    print(f"  + FLOPs: {_flops:.2f}G")
    """
    # print()
    
    print("* After compression w/ FCNN")
    print(f"  + Accuracy: {_acc:.2f} ({_acc-orig_acc:.2f})")
    print(f"  + Parameters: {_params:.2f}M ({orig_params/_params:.2f}x)")
    print(f"  + FLOPs: {_flops:.2f}G ({orig_flops/_flops:.2f}x)")
    
    print("="*30)
