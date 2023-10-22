import torch
from torch.autograd import Variable

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
