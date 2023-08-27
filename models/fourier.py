import numpy as np
import torch
import torch.nn as nn 
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
        # w_hat = fftn(_w, s=[self.cin, l_pad, l_pad])
        # self.weight = nn.Parameter(w_hat)        
        """
        self.weight = nn.Parameter(
                (torch.rand(self.n, self.cin//self.kappa , self.k, self.k) - 0.5) * 2 * _range
                )        
        """

    def forward(self, x):
        l_pad = self.l + self._pad
        t1 = time.time()
        x_hat = rfftn(x, s=[self.cin, l_pad, l_pad]) # input: [b, cin, l, l]
        t2 = time.time()
        # print(t2-t1)
        w_hat = rfftn(self.weight, s=[self.cin, l_pad, l_pad])
        # w_hat = self.weight
        freq_out = torch.einsum('bchw,nchw->bnchw', x_hat, w_hat)
        t3 = time.time()
        out = irfftn(freq_out, s=[l_pad, l_pad]) # .real[:,:,:,_pad:, _pad:] # b,n,c,h,w
        t4 = time.time()
        print("****** Type: ", out.dtype, self.wn.dtype)
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
        _w = torch.exp(torch.complex(real=torch.tensor([0.]), 
                                     imag=torch.tensor([(2*torch.pi)/(self.cin)])))
        wn = torch.pow(_w, _M) / self.cin # shape: [cin, cin//sr]
        return wn






