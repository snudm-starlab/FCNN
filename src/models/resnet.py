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
# resnet.py
# - implementation of ResNet18 and ResNet34 for CIFAR100
#
# This code is mainly based on the [GitHub Repository]
# [GitHub Repository]: https://github.com/weiaicunzai/pytorch-cifar100
################################################################################

import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    """ Convolution block for ResNet18 and ResNet34 """

    def __init__(self, in_channels, out_channels, stride=1):
        """
        an initialization function for BasicBlock

        * Inputs:
            - in_channels: the number of input channels
            - out_channels: the number of output channels
            - stride: stride for 2d convolutions
        """
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # shortcut
        self.shortcut = nn.Sequential()

        # use 1*1 convolution to match the dimension for shortcut
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        """
        a forward function for convolutional block

        * Inputs:
            - x: an input
        * Outputs:
            - out: an output
        """
        out = nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))
        return out

class ResNet(nn.Module):
    """ A ResNet model    """

    def __init__(self, block, num_blocks, num_classes=100):
        """
        an initialization function for ResNet

        * Inputs:
            - block: type of block
            - num_blocks: the number of blocks in each stage of ResNet
            - num_classes: the number of classes
        """
        super().__init__()
        
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True))
        # generate stages of ResNet
        self.conv2_x = self._make_stages(block, 64, num_blocks[0], 1)
        self.conv3_x = self._make_stages(block, 128, num_blocks[1], 2)
        self.conv4_x = self._make_stages(block, 256, num_blocks[2], 2)
        self.conv5_x = self._make_stages(block, 512, num_blocks[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_stages(self, block, out_channels, num_blocks, stride):
        """
        an initialization function for ResNet

        * Inputs:
            - block: type of block
            - out_channels: the number of output channels
            - num_blocks: the number of blocks in a stage of ResNet
            - stride: stride for 2d convolutions
        * Outputs:
            - stage: a Sequential module of a ResNet stage
        """
        
        strides = [stride] + [1] * (num_blocks - 1)
        # Append blocks in the stage
        layers = []
        for i, stride in enumerate(strides):                
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        stage = nn.Sequential(*layers)
        return stage

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

def resnet18():
    """
    returning a ResNet18 model

    * Outputs:
        - net: a Resnet18 object
    """
    net = ResNet(BasicBlock, [2, 2, 2, 2])
    return net
def resnet34():
    """
    returning a ResNet34 model

    * Outputs:
        - net: a Resnet34 object
    """
    net = ResNet(BasicBlock, [3, 4, 6, 3])
    return net 
