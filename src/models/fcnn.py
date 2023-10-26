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
# fcnn.py
# - implementation of FCNN verison of ResNet18 and ResNet34 for CIFAR100
#
# This code is mainly based on the [GitHub Repository]
# [GitHub Repository]: https://github.com/weiaicunzai/pytorch-cifar100
################################################################################


import torch
import torch.nn as nn
import time

class FConv3d(nn.Module):
    """ Flexibile Convolution (FConv) for FCNN """
    def __init__(self, kernel_size, in_channels, out_channels, stride,
                 nu=4, rho=4):
        """
        an initialization function for FConv3d

        * Inputs:
            - kernel_size: filter size for width an height of filters
            - in_channels: the number of input channels
            - out_channels: the number of output channels
            - stride: stride for 2d convolutions
            - rho: a coefficient for choosing kenel size for channel direction
            - nu: a coefficient for choosing the number of filters
        """
        super().__init__()
        self.cin = in_channels
        self.cout = out_channels
        self.k = kernel_size
        self.stride = stride
        self.rho = rho
        self.nu = nu

        # 3D convolution for implementing FConv
        self.conv = nn.Conv3d(1, out_channels//nu, (in_channels//rho, kernel_size, kernel_size), 
                         stride=(in_channels//rho, stride,stride), 
                                 padding= (0,self.k//2,self.k//2),
                                 padding_mode='zeros', bias=False)

        # for channel shuffling
        _inds = []
        _c = -1 
        for _n in range(nu):
            for _r in range(self.cout//nu):
                _c = (_c+1) % rho
                _inds.append(_r * rho + _c)
        self._inds = torch.tensor(_inds).cuda()

    def forward(self, x):
        """
        a forward function for FConv3d

        * Inputs:
            - x: an input
        * Outputs:
            - out: an output
        """
        bs, cin, L, _ = x.shape
        L_tilde = torch.div(L, self.stride, rounding_mode='floor')
        out = self.conv(x.view(bs, 1, cin, L, L)) # bs, cout//nu, rho, L, L 
        out = out.view(bs, -1, L_tilde, L_tilde)    
        # Channel selection
        out = out[:, self._inds , :,:] # (cin//nu*rho --> cin)
        return out

class FConv3dBlock(nn.Module):
    """ Convolution block for ResNet18 and ResNet34 """

    def __init__(self,in_channels, out_channels, stride=1,
                 kernel_size=3, nu=4, rho=4):
        """
        an initialization function for FConv3dBlock

        * Inputs:
            - in_channels: the number of input channels
            - out_channels: the number of output channels
            - stride: stride for 2d convolutions
            - kernel_size: filter size for width an height of filters
            - rho: a coefficient for choosing kenel size for channel direction
            - nu: a coefficient for choosing the number of filters
        """
        super().__init__()
        
        # residual function
        self.residual_function = nn.Sequential(
            FConv3d(kernel_size, in_channels, out_channels, 
                    stride=stride, rho=rho, nu=nu),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            FConv3d(kernel_size, out_channels, out_channels, 
                    stride=1, rho=rho, nu=nu),
            nn.BatchNorm2d(out_channels),
        )
        # shortcut
        self.shortcut = nn.Sequential()

        # use 1*1 convolution to match the dimension for shortcut
        if stride != 1:
            self.shortcut = nn.Sequential(
                FConv3d(1, in_channels, out_channels, 
                    stride=stride, rho=rho, nu=nu),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        """
        a forward function for FConv block

        * Inputs:
            - x: an input
        * Outputs:
            - out: an output
        """
        res1 = self.residual_function(x)
        res2 = self.shortcut(x)
        out= nn.ReLU(inplace=True)(res1 + res2)        
        return out

class FCNN(nn.Module):
    """ A FCNN model    """

    def __init__(self, block, num_blocks, num_classes=100, nu=16, rho=4):
        super().__init__()
        """
        an initialization function for FCNN

        * Inputs:
            - block: type of block
            - num_blocks: the number of blocks in each stage of ResNet
            - num_classes: the number of classes
            - rho: a coefficient for choosing kenel size for channel direction
            - nu: a coefficient for choosing the number of filters
        """

        self.in_channels = 64
        self.length = 32

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        # generate stages of ResNet
        self.conv2_x = self._make_stages(block, 64, num_blocks[0], 1, nu=nu, rho=rho)
        self.conv3_x = self._make_stages(block, 128, num_blocks[1], 2, nu=nu, rho=rho)
        self.conv4_x = self._make_stages(block, 256, num_blocks[2], 2, nu=nu, rho=rho)
        self.conv5_x = self._make_stages(block, 512, num_blocks[3], 2, nu=nu, rho=rho)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_stages(self, block, out_channels, num_blocks, stride, nu, rho):
        """
        an initialization function for ResNet

        * Inputs:
            - block: type of block
            - out_channels: the number of output channels
            - num_blocks: the number of blocks in a stage of ResNet
            - stride: stride for 2d convolutions
            - nu: a coefficient for choosing the number of filters
            - rho: a coefficient for choosing kenel size for channel direction
        * Outputs:
            - stage: a Sequential module of a FCNN stage
        """

        strides = [stride] + [1] * (num_blocks - 1)
        # Append blocks in the stage
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride=stride, 
                                kernel_size=3, nu=nu, rho=rho))
            self.in_channels = out_channels
        self.length = self.length // 2
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        a forward function for convolutional block

        * Inputs:
            - x: an input
        * Outputs:
            - out: an output
        """
        out = self.conv1(x)
        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = self.conv5_x(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def fcnn18(nu=6, rho=2):
    """
    returning a FCNN version of ResNet18 model

    * Outputs:
        - net: a FCNN object
    """
    net = FCNN(FConv3dBlock, [2, 2, 2, 2], nu=nu, rho=rho)
    return net

def fcnn34(nu=6, rho=2):
    """
    returning a FCNN version of ResNet34 model

    * Outputs:
        - net: a FCNN object
    """
    net = FCNN(FConv3dBlock, [3, 4, 6, 3], nu=nu, rho=rho)
    return net


