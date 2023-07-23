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
        self.w_dict = {} # (shape) -> cuda tensor 
        
        # Initialize parameters
        """
        _range = torch.sqrt(torch.tensor(1/(in_channels * kernel_size * kernel_size)))
        self.weight = nn.Parameter(
                        (torch.rand(out_channels, in_channels, kernel_size, kernel_size) 
                        - 0.5) * 2 * _range
                        )
        """
        # (out_channels, in_channels, 7) # R0, Rr, Rc, Rd
        # self.weight = nn.Parameter(torch.rand(out_channels, in_channels, 7))
        self.wr = nn.Parameter(torch.rand(out_channels, in_channels, L+self.k//2,L+self.k//2))
        self.wi = nn.Parameter(torch.rand(out_channels, in_channels, L+self.k//2,L+self.k//2))
        # self.wi = None

    def conv2fconv(self, _conv):
        _weight = _conv.weight
        _pad = self.k // 2
        _w_hat = rfftn(_weight.data.flip(-2).flip(-1), s=(self.L+_pad, self.L+_pad))
        _wr, _wi = _w_hat.real, _w_hat.imag
        """ 
        print("wr.shape: ", _wr.shape)
        print("Slice.shape: ", _wr[:, :, :self.L//2, 1:].mean((2,3)).shape) 
        print("self.cout: ", self.cout, "self.cin", self.cin)
        print("Slice.shape: ", _wr[:, :, :self.L//2, 1:].mean((2,3)).view(self.cout,self.cin,1).shape) 
        print("L->", self.L)
        print("Nan?->", _wi[:, :, 1:self.L//2+1, 1:self.L//4+1])
        """
        """
        self.weight[:,:,0] = _wr[:, :, 0, 0]
        self.weight[:,:,1] = _wr[:, :, :self.L//2+1, 1:].mean((2,3))
        self.weight[:,:,2] = _wr[:, :, 1:, :self.L//4+1].mean((2,3))
        self.weight[:,:,3] = _wr[:, :, 1:self.L//2+1, 1:self.L//4+1].mean((2,3))
        self.weight[:,:,4] = _wi[:, :, :self.L//2+1, 1:].mean((2,3))
        self.weight[:,:,5] = _wi[:, :, 1:, :self.L//4+1].mean((2,3))
        self.weight[:,:,6] = _wi[:, :, 1:self.L//2+1, 1:self.L//4+1].mean((2,3))
        # print(self.weight[0,2,:])
        print(self.weight[:,:,1].shape)
        """
        self.wr.data = _wr
        self.wi.data = _wi


                
    def forward(self, x):
        # print("Hello~ Let's forward!")
        bs, cin, L = x.shape[0], x.shape[1], x.shape[2]
        _pad = self.k // 2
        # _pad = 1
        # LL = 1
        LL = L+_pad
        
        st = time.time()
        x_hat = rfftn(x, s=(LL, LL)) # x = [bs, cin, H, W]
        # print("* Time for input dft: ", time.time()-st)

        st = time.time()
        # w_hat = rfftn(self.weight.data.flip(-2).flip(-1), s=(LL, LL))
        # w_hat = 
        """
        _key = (self.cout, self.cin, LL, int(LL/2)+1)
        if _key in self.w_dict:
            wr = torch.zeros_like(self.w_dict[_key])
        else:
            wr = torch.zeros(self.cout, self.cin, LL, int(LL/2)+1).cuda()
            self.w_dict[_key] = wr

        wi = torch.zeros_like(wr)
        wr[:,:,0,0] += self.weight[:,:,0]
        wr[:,:,:self.L//2+1, 1:] += self.weight[:,:,1].view(self.cout,self.cin,1,1)
        wr[:,:,1:, :self.L//4+1] += self.weight[:,:,2].view(self.cout,self.cin,1,1)
        wr[:,:,1:self.L//2+1, 1:self.L//4+1] += (self.weight[:,:,3]-self.weight[:,:,1]-self.weight[:,:,2])\
                .view(self.cout,self.cin,1,1)
        wi[:,:,:self.L//2+1, 1:] += self.weight[:,:,4].view(self.cout,self.cin,1,1)
        wi[:,:,1:, :self.L//4+1] += self.weight[:,:,5].view(self.cout,self.cin,1,1)
        wi[:,:,1:self.L//2+1, 1:self.L//4+1] += (self.weight[:,:,6]-self.weight[:,:,4]-self.weight[:,:,5])\
                .view(self.cout,self.cin,1,1)
        """
        """
        for _r in range(LL):
            for _c in range(LL//2 + 1):
                if _r == 0 and _c ==0:
                    wr[:,:,_r,_c] += self.weight[:,:,0]
                elif 0 < _r and _r <L//2+1 and 0 < _c and _c < L//4+1:
                    wr[:,:,_r,_c] += self.weight[:,:,3]
                    wi[:,:,_r,_c] += self.weight[:,:,6]
                elif 0 < _r and _c < L//4+1:
                    wr[:,:,_r,_c] += self.weight[:,:,2]
                    wi[:,:,_r,_c] += self.weight[:,:,5]
                elif _r < L//2 + 1 and 0 < _c:
                    wr[:,:,_r,_c] += self.weight[:,:,1]
                    wi[:,:,_r,_c] += self.weight[:,:,4]
        """
        # thr = 2e-2
        # self.wr[self.wr.abs() < thr] = 0.
        # self.wi[self.wi.abs() < thr] = 0.
        # print('-> ', (self.wr.abs() < thr).sum()/self.wr.numel() * 100)


        w_hat = torch.complex(self.wr, self.wi)

        # print("* Time for weight dft: ", time.time()-st)
        
        st = time.time()
        res = torch.einsum('bihw,oihw->bohw', x_hat, w_hat) # [bs, cout, h, w]
        # print("* Time for matmul: ", time.time()-st)
        
        # print("**** Res.shape -> ", res.shape)
        st = time.time()
        out = irfftn(res, s=(L+_pad,L+_pad)).real[:,:,_pad:, _pad:]
        # print("* Time for idft: ", time.time()-st)
        # print("")

        if self.stride==2:
            _inds = torch.arange(L)[torch.arange(L)%2==0].to(x.device)
            out = torch.index_select(out, 2, _inds)
            out = torch.index_select(out, 3, _inds)
        return out

    def analyze(self,):
        _pad = self.k//2
        wh = rfftn(self.weight.data.flip(-2).flip(-1), s=(self.L+_pad, self.L+_pad))
        tote = (wh.real**2 + wh.imag**2).sum()
        # re = (wh.real**2 + wh.imag**2).sum(0).sum(0) / tote * 100
        # ri = (wh.imag**2).sum(0).sum(0) / tote * 100
        # re = (wh.real**2).sum(0).sum(0)
        # ri = (wh.imag**2).sum(0).sum(0)
        re = (wh.real**2).mean((0,1))
        ri = (wh.imag**2).mean((0,1))

        # print()
        # print(f"* Real: {wh.real.shape}\n", re)
        # print(f"* Imag: {wh.imag.shape}\n", ri)
        # print()



# Conver 2d cnn -> Fourier conv layer
@torch.no_grad()
def convert(net, layer, block, conv_num):
    _conv = getattr(getattr(net, f'conv{layer}_x'), f'{block}').residual_function[conv_num]
    c_out, c_in, ks, _ = _conv.weight.shape
    stride = _conv.stride[0]
    L_dict = {1:32, 2:32, 3:16, 4:8, 5:4}
    L = L_dict[layer]
    if layer > 2 and block==0 and conv_num == 0:
        L *= 2
    f_conv = FConv2d(c_in, c_out, stride, ks, L)
    f_conv.conv2fconv(_conv)
    # f_conv.weight.data = _conv.weight.clone().detach()
    # print(f"[Convert] Layer {layer} | block {block} | conv_num {conv_num}")
    # f_conv.analyze()
    getattr(getattr(net, f'conv{layer}_x'), f'{block}').residual_function[conv_num] = f_conv
    return net


def reg(net, reg_type):
    res = torch.tensor(0.).cuda()
    for _n, _p in net.named_parameters():
        if 'wr' in _n or 'wi' in _n:
            if reg_type == 'l1':
                res += _p.abs().sum()
            else:
                res += (_p*_p).sum()

    return res

def zero_out(net, thr):
    with torch.no_grad():
        for _n, _p in net.named_parameters():
            if 'wr' in _n or 'wi' in _n:
                _p[_p.abs()<thr] = 0.

def count_params(net, thr):
    _num = 0
    for _n, _p in net.named_parameters():
        if 'wr' in _n or 'wi' in _n:
            _num += (_p.abs()>thr).sum()
            # print("_n: ", _n)
        else:
            _num += _p.numel()
    return _num




