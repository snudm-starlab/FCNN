""" helper function

author baiyu
"""
import os
import sys
import re
import datetime

from torch.autograd import Variable
import numpy

import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_network(args):
    """ return given network
    """
    if args.net == 'resnet34':
        from models.resnet import resnet34
        net = resnet34()

    elif args.net == 'mobilenetv2':
        from models.mobilenetv2 import mobilenetv2
        net = mobilenetv2()

    elif args.net == 'dmobilenetv2':
        from models.mobilenetv2 import mobilenetv2
        net = mobilenetv2(dfirst=True)
    
    elif args.net == 'fmobilenetv2':
        from models.mobilenetv2 import fmobilenetv2
        net = fmobilenetv2(dfirst=False, rho=args.rho, nu=args.nu)
    
    elif args.net == 'dfmobilenetv2':
        from models.mobilenetv2 import fmobilenetv2
        net = fmobilenetv2(dfirst=True, rho=args.rho, nu=args.nu)
 
    elif args.net == 'fcnn34':
        from models.fcnn import fcnn34
        net = fcnn34(nu=args.nu, rho=args.rho)

    elif args.net == 'dscresnet34':
        from models.dsc_resnet import dscresnet34
        net = dscresnet34()

    elif args.net == 'gcresnet34':
        from models.gc_resnet import gcresnet34
        net = gcresnet34(nu=args.nu)

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu: #use_gpu
        net = net.cuda()

    return net


def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_training = CIFAR100Train(path, transform=transform_train)
    cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader

def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_test = CIFAR100Test(path, transform=transform_test)
    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader

def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = numpy.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = numpy.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def most_recent_folder(net_weights, fmt):
    """
        return most recent created folder under net_weights
        if no none-empty folder were found, return empty folder
    """
    # get subfolders in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''

    # sort folders by folder created time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]

def most_recent_weights(weights_folder):
    """
        return most recent created weights file
        if folder is empty return empty string
    """
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    # sort files by epoch
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files[-1]

def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
       raise Exception('no recent weights were found')
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch

def best_acc_weights(weights_folder):
    """
        return the best acc .pth file in given folder, if no
        best acc weights file were found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == 'best']
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return best_files[-1]

def get_flops(net, args, imagenet=False):
    """
    Counting number of parameters in the model
    
    :param net: the model to be calculated for parameters and flops
    :param imagenet: which dataset is used
    """

    multiply_adds = True
    list_conv3d = []

    def conv3d_hook(self, _input, output):
        """
        Hook of convolution layer
        Calculate the FLOPs of the convolution layer
        
        :param input: input size
        :param output: output size
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
        Hook of convolution layer
        Calculate the FLOPs of the convolution layer
        
        :param input: input size
        :param output: output size
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
        Hook of linear layer
        Calculate the FLOPs of the linear layer
        
        :param input: input size
        :param output: output size
        """
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()

        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)

    list_bn = []

    def bn_hook(self, input, output):
        """
        Hook of batch normalization(BN) layer
        Calculate the FLOPs of the BN layer
        
        :param input: input size
        :param output: output size
        """
        list_bn.append(input[0].nelement())

    list_relu = []

    def relu_hook(self, input, output):
        """
        Hook of ReLU layer
        Calculate the FLOPs of the ReLU layer
        
        :param input: input size
        :param output: output size
        """
        list_relu.append(input[0].nelement())

    list_pooling = []

    def pooling_hook(self, input, output):
        """
        Hook of pooling layer
        Calculate the FLOPs of the pooling layer
        
        :param input: input size
        :param output: output size
        """
        batch_size, _, _, _ = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width

        list_pooling.append(flops)

    def foo(net):
        """
        Calculate the total number of FLOPs of a net
        
        :param net: net to be calculated
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
            foo(c)

    foo(net)
    if imagenet:
        input = Variable(torch.rand(3, 224, 224).unsqueeze(0), requires_grad=True)
    else:
        input = Variable(torch.rand(3, 32, 32).unsqueeze(0), requires_grad=True).cuda()
    _ = net(input)
    total_flops = (sum(list_conv3d) + sum(list_conv) + sum(list_linear)
                   + sum(list_bn) + sum(list_relu) + sum(list_pooling))

    return total_flops / 1e9
