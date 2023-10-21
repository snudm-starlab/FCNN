"""resnet in pytorch



[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

from re import I
import torch
import torch.nn as nn


class CustomConv3d(nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels, stride, bias=False,
                 kappa=4, nu=4):
        super().__init__()
        self.cin = in_channels
        self.cout = out_channels
        self.k = kernel_size
        self.stride = stride
        self.kappa = kappa
        self.nu = nu
        # c_pad = (in_channels//kappa - in_channels//nu)//2+1 if kappa < nu else 0
        self.conv = nn.Conv3d(1, out_channels//nu, (in_channels//kappa, kernel_size, kernel_size), 
                         stride=(in_channels//kappa, stride,stride), padding= (0,self.k//2,self.k//2),
                               padding_mode='zeros', bias=bias)
        ratio = kappa//nu
        _inds = []
        _c = -1 
        for _n in range(nu):
            for _r in range(self.cout//nu):
                _c = (_c+1) % kappa
                _inds.append(_r * kappa + _c)
        self._inds = torch.tensor(_inds).cuda()
        """
        ratio = kappa//nu
        self.channel_map = torch.zeros(kappa, self.cout//nu) # select kappa -> nu

        for r in range(kappa):
            for c in range(self.cout//nu):
                if (r+c)%(ratio)==0:
                    self.channel_map[r][c] = 1
        self.channel_map = self.channel_map.bool().cuda()
        """

    def forward(self, x):
        bs, cin, L, _ = x.shape
        L_tilde = L // self.stride
        out = self.conv(x.view(bs, 1, cin, L, L)) # bs, cout//nu, kappa, L, L 
        out = out.view(bs, -1, L_tilde, L_tilde)    
        # print("Before selection: ", out.shape)
        # Channel Selection
        out = out[:, self._inds , :,:] # (cin//nu*kappa --> cin)
        # print("After selection: ", out.shape)
        return out
    """
    def forward(self, x):
        bs, cin, L, _ = x.shape
        L_tilde = L // self.stride
        out = self.conv(x.view(bs, 1, cin, L, L)) # bs, cout//nu, kappa, L, L 
        # Channel shuffling
        # (bs, cout//nu, kappa, L, L) --> (bs, kappa, cout//nu, L, L)
        out = out.transpose(1,2).contiguous() 
        # (bs, kappa, cout//nu, L, L) --> (bs, nu, cout//nu, L, L)
        out = out[:,self.channel_map, :,:] # cin --> nu
        out = out.view(bs, -1, L_tilde, L_tilde)    
        return out
    """

class Conv3dBasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self,in_channels, out_channels, stride=1,
                 kernel_size=3, kappa=4, nu=4):
        super().__init__()
        
        #residual function
        self.residual_function = nn.Sequential(
            CustomConv3d(kernel_size, in_channels, out_channels, 
                    stride=stride, kappa=kappa, nu=nu),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            CustomConv3d(kernel_size, out_channels, out_channels, 
                    stride=1, kappa=kappa, nu=nu),
            # FConv2d(L, C*N, 1, stride=1),
            nn.BatchNorm2d(out_channels),
        )
        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1:
            self.shortcut = nn.Sequential(
                # FConv2d(length*stride, in_channels, out_channels, num_filters, 
                #         stride=stride, kernel_size=1),
                # CustomConv3d(1, in_channels, out_channels, 
                #     stride=stride, kappa=kappa, nu=nu),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        res1 = self.residual_function(x)
        # print("_____________________________ Beftore Shrotcut Ends ! ______________")
        res2 = self.shortcut(x)

        # print("* Residual: ", res1.shape, " / shortcut: ", res2.shape)
        
        # out= nn.ReLU(inplace=True)(self.residual_function(x)
        #                              + self.shortcut(x))
        out= nn.ReLU(inplace=True)(res1 + res2)
        
        """
        xx = (self.fc1(x) + self.fc2(x) + self.fc3(x))/3
        out = (nn.ReLU(inplace=True)(self.fc4(xx) + self.shortcut(x)) + 
              nn.ReLU(inplace=True)(self.fc5(xx) + self.shortcut(x)) + 
              nn.ReLU(inplace=True)(self.fc6(xx) + self.shortcut(x))) / 3
        """
        return out

class Conv3dBottleNeck(nn.Module):
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

class Conv3dResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=100, nu=16, kappa=4):
        super().__init__()

        self.in_channels = 64
        self.length = 32

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
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
            layers.append(block(self.in_channels, out_channels, stride=stride, 
                                kernel_size=3, kappa=kappa, nu=nu))
            self.in_channels = out_channels * block.expansion
        self.length = self.length // 2
        # def __init__(self,in_channels, out_channels, stride=1,
        #          kernel_size=3, kappa=4, nu=4):
        
        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        # print("**************** Conv1 Ends!")
        output = self.conv2_x(output)
        # print("**************** Conv2 Ends!")
        output = self.conv3_x(output)
        # print("**************** Conv3 Ends!")
        output = self.conv4_x(output)
        # print("**************** Conv4 Ends!")
        output = self.conv5_x(output)
        # print("**************** Conv5 Ends!")
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output

def conv3dfresnet18():
    """ return a ResNet 18 object
    """
    return Conv3dResNet(Conv3dBasicBlock, [2, 2, 2, 2])

def conv3dresnet34(nu=6, kappa=2):
    """ return a ResNet 34 object
    """
    return Conv3dResNet(Conv3dBasicBlock, [3, 4, 6, 3], nu=nu, kappa=kappa)
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


