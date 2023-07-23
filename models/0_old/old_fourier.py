import torch
import torch.nn as nn 
from torch.fft import fft2, ifft2, fftn, ifftn, rfftn, irfftn
import torch.nn.functional as F
import time

# Fourier conv layer
class FConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_size, L):
        # Omit bias for simplicity
        super().__init__()
        # Set variables 
        self.cout = out_channels
        self.cin = in_channels
        self.stride = stride
        self.k = kernel_size
        self.L = L
        
        # Initialize parameters
        _range = torch.sqrt(torch.tensor(1/(in_channels * kernel_size * kernel_size)))
        self.weight = nn.Parameter(
                        (torch.rand(out_channels, in_channels, kernel_size, kernel_size) 
                        - 0.5) * 2 * _range
                        )

    def forward(self, x):
        bs, L = x.shape[0], x.shape[2]
        _pad = self.k // 2
        # _pad = 0
        """
        padded_x = torch.zeros(x.shape[0], self.cout, x.shape[1], 
                               x.shape[2]+_pad, x.shape[3]+_pad).to(x.device)
        for _c in range(self.cout):
            padded_x[:x.shape[0], _c:_c+1,  :x.shape[1], :x.shape[2], :x.shape[3]] \
                    += x.view(bs,1, self.cin, L, L)
        x_hat_orig = fft2(padded_x)
        """
        
        # padded_x = F.pad(x, (0,_pad,0,_pad), 'constant', 0)
        x_hat = rfftn(x, s=(L+_pad, L+_pad)) # x = [bs, cin, H, W]
        # print("x_hat: ", x_hat.shape)
        st = time.time()
        # x_hat = x_hat.repeat([1,self.cout,1,1]).view(
        #         bs, self.cout, self.cin, x_hat.shape[-2], x_hat.shape[-1])
        # print("* Time for input dft: ", time.time()-st)

        st = time.time()
        w_hat = rfftn(self.weight.data.flip(-2).flip(-1), s=(L+_pad, L+_pad))
        # print("* Time for weight dft: ", time.time()-st)
        # w_hat = w_hat.repeat([bs,1,1,1]).view(bs,self.cout, self.cin, 
        #                                             w_hat.shape[-2], w_hat.shape[-1])
        
        st = time.time()
        # res = torch.einsum('bihw,oihw->bohw', x_hat, w_hat)
        # res = (x_hat * w_hat)[:,:,:,:,0:3].sum(dim=2) # [bs, cout, cin, h, w]
        # res = (x_hat * w_hat).sum(dim=2) # [bs, cout, cin, h, w]
        # print("* Time for matmul: ", time.time()-st)
        
        # print("**** Res.shape -> ", res.shape)
        st = time.time()
        out = irfftn(res, s=(L+_pad,L+_pad)).real[:,:,_pad:, _pad:]
        # out = irfftn(res, s=(L+_pad,L+_pad)).real.sum(dim=2)[:,:,_pad:, _pad:]
        # print("* Time for idft: ", time.time()-st)
        # print("")

        if self.stride==2:
            _inds = torch.arange(L)[torch.arange(L)%2==0].to(x.device)
            out = torch.index_select(out, 2, _inds)
            out = torch.index_select(out, 3, _inds)
        return out

    def analyze(self,):
        _pad = self.k//2
        wh = rfftn(self.weight, s=(self.L+_pad, self.L+_pad))
        tote = (wh.real**2 + wh.imag**2).sum()
        # re = (wh.real**2 + wh.imag**2).sum(0).sum(0) / tote * 100
        # ri = (wh.imag**2).sum(0).sum(0) / tote * 100
        # re = (wh.real**2).sum(0).sum(0)
        # ri = (wh.imag**2).sum(0).sum(0)
        re = (wh.real**2).mean((0,1))
        ri = (wh.imag**2).mean((0,1))

        # print()
        print(f"* Real: {wh.real.shape}\n", re)
        print(f"* Imag: {wh.imag.shape}\n", ri)
        print()



# Conver 2d cnn -> Fourier conv layer
def convert(net, layer, block, conv_num):
    _conv = getattr(getattr(net, f'conv{layer}_x'), f'{block}').residual_function[conv_num]
    c_out, c_in, ks, _ = _conv.weight.shape
    stride = _conv.stride[0]
    L_dict = {1:32, 2:32, 3:16, 4:8, 5:4}
    L = L_dict[layer]
    if layer > 2 and block==0 and conv_num == 0:
        L *= 2
    f_conv = FConv2d(c_in, c_out, stride, ks, L)
    f_conv.weight.data = _conv.weight.clone().detach()
    print(f"* Layer {layer} | block {block} | conv_num {conv_num}")
    f_conv.analyze()
    getattr(getattr(net, f'conv{layer}_x'), f'{block}').residual_function[conv_num] = f_conv
    return net

