import numpy as np
import torch
import torch.nn as nn 
from torch.fft import fft2, ifft2, fftn, ifftn, rfftn, irfftn
import torch.nn.functional as F
import time


class FConv2d(torch.nn.Module):
    def __init__(self, length, in_channels, out_channels, num_filters, stride,
                 kernel_size=None):
        super().__init__()
        self.cin = in_channels
        self.cout = out_channels
        self.l = length
        self.n = num_filters
        self.s = stride
        
        if kernel_size is not None:
            self.k = kernel_size
        else:
            if self.l > 16:
                self.k = self.l//4 + 1
            else:
                self.k = self.l//2 + 1

        _range = torch.sqrt(torch.tensor(1/(self.cin * self.k * self.k)))
        self.weight = nn.Parameter(
                (torch.rand(self.n, self.cin//8 , self.k, self.k) - 0.5) * 2 * _range
                )


    def forward(self, x):
        x_hat = rfftn(x, dim=[1,2,3]) # input: [b, cin, l, l]
        w_hat = rfftn(self.weight, s=[self.cin, self.l, self.l] )
        # print("x_hat.shape: ", x_hat.shape, "w_hat.shape: ", w_hat.shape)
        # freq_out = torch.einsum('bchw,nchw->bnchw', x_hat, w_hat)
        # out = irfftn(freq_out, dim=[2,3,4]).real # b,n,c,h,w
        freq_out = torch.einsum('bchw,nchw->bnchw', x_hat, w_hat)
        out = irfftn(freq_out, dim=[2,3,4]).real # b,n,c,h,w

        # channel shuffling
        shrink_ratio = (self.n * self.cin) // (self.cout)
        _inds = torch.arange(self.cin)[torch.arange(self.cin)%(shrink_ratio)==0].to(x.device)
        out = torch.index_select(out, 2, _inds) # [b, n, c//n, h, w]
        out = out.view(out.shape[0], -1, out.shape[3], out.shape[4]) # [b,c,h,w]
                
        if self.s==2:
            _inds = torch.arange(self.l)[torch.arange(self.l)%2==0].to(x.device)
            out = torch.index_select(out, 2, _inds)
            out = torch.index_select(out, 3, _inds)
        # print("Out (stride): ", out.shape)
        
        return out






