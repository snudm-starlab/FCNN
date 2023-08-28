"""resnet in pytorch



[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn
import numpy as np
from torch.fft import fft2, ifft2, fftn, ifftn, rfftn, irfftn
import torch.nn.functional as F
import time


class FConv2d(torch.nn.Module):
    def __init__(self, length, in_channels, out_channels, num_filters, stride,
                 kernel_size=None, kappa=4):
        super().__init__()
        self.cin = in_channels
        self.cout = out_channels
        self.l = length
        self.n = num_filters
        self.s = stride
        self.kappa = kappa
        
        
        kernel_size=3
        self.k = kernel_size
        self.wn = self.get_wn().cuda()
        """
        if kernel_size is not None:
            self.k = kernel_size
        else:
            if self.l > 16:
                self.k = self.l//4 + 1
            else:
                self.k = self.l//2 + 1
        """
        _range = torch.sqrt(torch.tensor(self.kappa/(self.cin * self.k * self.k)))
        self._pad = 2 # self.k//2
        l_pad = self.l + self._pad
        _w = (torch.rand(self.n, self.cin//self.kappa , self.k, self.k) - 0.5) * 2 * _range
        self.weight = nn.Parameter(_w)

    def forward(self, x):
        l_pad = self.l + self._pad
        t1 = time.time()
        x_hat = fftn(x, s=[self.cin, l_pad, l_pad]) # input: [b, cin, l, l]
        t2 = time.time()
        # print(t2-t1)
        w_hat = fftn(self.weight, s=[self.cin, l_pad, l_pad])
        # w_hat = self.weight
        freq_out = torch.einsum('bchw,nchw->bnchw', x_hat, w_hat)
        t3 = time.time()
        out = ifftn(freq_out, s=[l_pad, l_pad]) # .real[:,:,:,_pad:, _pad:] # b,n,c,h,w
        # out = torch.complex(real=out, imag=torch.zeros_like(out))
        t4 = time.time()
        # print("****** Type: ", out.dtype, self.wn.dtype)
        out = torch.einsum("cd, bnchw->bndhw", self.wn, out)
        t5 = time.time()
        out = out.real[:,:,:,self._pad:, self._pad:] # b,n,c,h,w

        # print(t2-t1, t3-t2, t4-t3, t5-t4)
        # raise Exception

        # channel shuffling
        # out = torch.index_select(out, 2, self._inds) # [b, n, c//n, h, w] # removing channels
        out = out.transpose(1,2).contiguous() # shuffling
        out = out.view(out.shape[0], -1, out.shape[3], out.shape[4]) # [b,c,h,w]
 
        if self.s==2:
            _inds = torch.arange(self.l)[torch.arange(self.l)%2==0].to(x.device)
            out = torch.index_select(out, 2, _inds)
            out = torch.index_select(out, 3, _inds) 
        return out

    def get_wn(self,):
        _v = torch.arange(self.cin).view(self.cin,-1)
        _sr = (self.n * self.cin) // (self.cout) # shrink ratio
        _inds = torch.arange(self.cin)[torch.arange(self.cin)%(_sr)==0]
        _M = _v @ _v[_inds,:].T
        _ww = torch.exp(torch.complex(real=torch.tensor([0.]), 
                                     imag=torch.tensor([(2*torch.pi)/(self.cin)])))
        wn = torch.pow(_ww, _M) / self.cin # shape: [cin, cin//sr]
        return wn


class FourierBasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, length, in_channels, out_channels, num_filters, stride=1,
                 kernel_size=None, kappa=4):
        super().__init__()
        
        #residual function
        self.residual_function = nn.Sequential(
            FConv2d(length*stride, in_channels, out_channels, num_filters, 
                    stride=stride, kappa=kappa),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            FConv2d(length, out_channels, out_channels, num_filters, 
                    stride=1, kappa=kappa),
            # FConv2d(L, C*N, 1, stride=1),
            nn.BatchNorm2d(out_channels),
        )
        #shortcut
        self.shortcut = nn.Sequential()
        self.stride = stride
        self.length = length

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        # if stride != 1:
        if out_channels != in_channels:
            self.shortcut = nn.Sequential(
                FConv2d(length*stride, in_channels, out_channels, num_filters, 
                        stride=stride, kernel_size=1),
                # nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        res1 = self.residual_function(x)
        res2 = self.shortcut(x)

        # print("* Residual: ", res1.shape, " / shortcut: ", res2.shape)
        
        # out= nn.ReLU(inplace=True)(self.residual_function(x)
        #                              + self.shortcut(x))
        # print("* Self.stride: ", self.stride, " | self.length: ", self.length)
        # print(res1.shape, res2.shape)
        out= nn.ReLU(inplace=True)(res1 + res2)
        
        """
        xx = (self.fc1(x) + self.fc2(x) + self.fc3(x))/3
        out = (nn.ReLU(inplace=True)(self.fc4(xx) + self.shortcut(x)) + 
              nn.ReLU(inplace=True)(self.fc5(xx) + self.shortcut(x)) + 
              nn.ReLU(inplace=True)(self.fc6(xx) + self.shortcut(x))) / 3
        """
        return out

class FourierBottleNeck(nn.Module):
    # TODO: Unimplemented
    """Residual block for resnet over 50 layers

    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class FourierResNetv2(nn.Module):

    def __init__(self, block, num_block, num_classes=100, nu=16, kappa=4):
        super().__init__()

        self.in_channels = 64
        self.length = 30

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=(0,0), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1, nu=nu, kappa=kappa)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2, nu=nu, kappa=kappa)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2, nu=nu, kappa=kappa)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2, nu=nu, kappa=kappa)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride, nu, kappa):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        # def __init__(self, L, C, N, stride=1):
        for stride in strides:
            layers.append(block(self.length, self.in_channels, out_channels, 
                                out_channels // nu, stride, kernel_size=None,
                                kappa=kappa))
            self.in_channels = out_channels * block.expansion
        if self.length > 6:
            self.length = (self.length+2) // 2 - 2
        # def __init__(self, length, in_channels, out_channels, num_filters, stride=1):
        #     super().__init__()
        
        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output
'''
def fresnet18():
    """ return a ResNet 18 object
    """
    return FourierResNet(FourierBasicBlock, [2, 2, 2, 2])
'''

def fresnetv2_34(nu=4, kappa=4):
    """ return a ResNet 34 object
    """
    return FourierResNetv2(FourierBasicBlock, [3, 4, 6, 3], nu=nu, kappa=kappa)
'''
def resnet50():
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3])

def resnet101():
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3])

def resnet152():
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3])
'''


