"""resnet in pytorch



[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

from re import I
import torch
import torch.nn as nn

class ChannelShuffle(nn.Module):

    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        batchsize, channels, height, width = x.data.size()
        # print(channels, channels.item())
        # channels_per_group = int(channels / self.groups)
        channels_per_group = channels // self.groups

        #"""suppose a convolutional layer with g groups whose output has
        #g x n channels; we first reshape the output channel dimension
        #into (g, n)"""
        x = x.view(batchsize, self.groups, channels_per_group, height, width)

        #"""transposing and then flattening it back as the input of next layer."""
        x = x.transpose(1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)

        return x

class DepthwiseConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, **kwargs),
            nn.BatchNorm2d(output_channels)
        )

    def forward(self, x):
        return self.depthwise(x)

class PointwiseConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, **kwargs):
        super().__init__()
        self.pointwise = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, **kwargs),
            nn.BatchNorm2d(output_channels)
        )

    def forward(self, x):
        return self.pointwise(x)

class GC2d(nn.Module):

    def __init__(self, input_channels, output_channels, stage, stride, nu):
        super().__init__()

        #"""Similar to [9], we set the number of bottleneck channels to 1/4
        #of the output channels for each ShuffleNet unit."""
        self.bottlneck = nn.Sequential(
            PointwiseConv2d(
                input_channels,
                int(output_channels / 4),
                # output_channels,
                groups=nu
            ),
            nn.ReLU()
        )

        #"""Note that for Stage 2, we do not apply group convolution on the first pointwise
        #layer because the number of input channels is relatively small."""
        if stage == 2:
            self.bottlneck = nn.Sequential(
                PointwiseConv2d(
                    input_channels,
                    int(output_channels / 4),
                    # output_channels,
                    groups=nu
                ),
                nn.ReLU()
            )

        self.channel_shuffle = ChannelShuffle(nu)

        self.depthwise = DepthwiseConv2d(
            # output_channels,
            # output_channels,
            int(output_channels / 4),
            int(output_channels / 4),
            3,
            groups=int(output_channels / 4),
            groups=output_channels,
            stride=stride,
            padding=1
        )

        self.expand = PointwiseConv2d(
            int(output_channels / 4),
            output_channels,
            # output_channels,
            groups=nu
        )

        # self.relu = nn.ReLU()

    def _add(self, x, y):
        return torch.add(x, y)

    def _cat(self, x, y):
        return torch.cat([x, y], dim=1)

    def forward(self, x):
        # shortcut = self.shortcut(x)

        shuffled = self.bottlneck(x)
        shuffled = self.channel_shuffle(shuffled)
        shuffled = self.depthwise(shuffled)
        shuffled = self.expand(shuffled)

        # output = self.fusion(shortcut, shuffled)
        # output = self.relu(output)
        # output = self.relu(shuffled)
        output = shuffled
        return output

class GCBasicBlock(nn.Module):
    """
    Basic Block for resnet 18 and resnet 34
    """
    expansion = 1

    def __init__(self,in_channels, out_channels, stride=1,
                 kernel_size=3, nu=4):
        super().__init__()
        
        #residual function
        self.residual_function = nn.Sequential(
            GC2d(in_channels, out_channels, kernel_size, 
                    stride=stride, nu=nu),
            nn.ReLU(inplace=True),
            GC2d(out_channels, out_channels, kernel_size,
                    stride=1, nu=nu),
        )
        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        res1 = self.residual_function(x)
        res2 = self.shortcut(x)        
        out= nn.ReLU(inplace=True)(res1 + res2)
        
        return out

class GCResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=100, nu=4):
        super().__init__()

        self.in_channels = 64
        self.length = 32

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1, nu=nu)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2, nu=nu)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2, nu=nu)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2, nu=nu)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride, nu):
        """
        make resnet layers(by layer i didnt mean this 'layer' was the
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
            layers.append(block(self.in_channels, out_channels, 
                                stride=stride, kernel_size=3, nu=nu))
            self.in_channels = out_channels * block.expansion
        self.length = self.length // 2 
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

def gcfresnet18():
    """ return a GCResNet 18 object
    """
    return GCResNet(GCBasicBlock, [2, 2, 2, 2])

def gcresnet34(nu):
    """ return a GCResNet 34 object
    """
    return GCResNet(GCBasicBlock, [3, 4, 6, 3], nu=nu)

def gcresnet50(nu):
    """ return a GCResNet 50 object
    """
    return GCResNet(GCBasicBlock, [3, 4, 6, 3], nu=nu)

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


