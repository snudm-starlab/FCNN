"""
Starlab CNN Compression with FCNN (Flexible Convolutional Neural Network)

unit_test.py
- Content
"""

import unittest

import torch
from torch import nn

from models.fcnn import FConv3d, FConv3dBlock, fcnn34
from models.resnet import resnet34

class FCNNTest(unittest.TestCase):
    """ A class for unittest of FCNN """

    # tests for FConv3d
    def test_fconv3d_num_channels(self):
        """Test if the number of channels is the same"""
        k=3
        cin=32
        cout=64
        stride=1
        nu_param=2
        rho=6
        dummy_input = torch.randn(16, cin, 32, 32).to(DEVICE)
        fconv3d = FConv3d(k,cin,cout,stride,nu_param,rho).to(DEVICE)
        conv2d = nn.Conv2d(cin,cout,k,stride=stride, padding='same').to(DEVICE)
        self.assertEqual(fconv3d(dummy_input).shape[1], conv2d(dummy_input).shape[1])
        self.assertEqual(fconv3d(dummy_input).shape[1], cout)

    def test_fconv3d_padding_test(self):
        """Test if the padding is the same"""
        k=3
        cin=32
        cout=64
        stride=1
        nu_param=2
        rho=6
        dummy_input = torch.randn(16, cin, 32, 32).to(DEVICE)
        fconv3d = FConv3d(k,cin,cout,stride,nu_param,rho).to(DEVICE)
        self.assertEqual(fconv3d(dummy_input).shape[2], dummy_input.shape[2])
        self.assertEqual(fconv3d(dummy_input).shape[3], dummy_input.shape[3])

    def test_fconv3d_stride_test(self):
        """Test if the stride is the same"""
        k=3
        cin=32
        cout=64
        stride=2
        nu_param=2
        rho=6
        dummy_input = torch.randn(16, cin, 32, 32).to(DEVICE)
        fconv3d = FConv3d(k,cin,cout,stride,nu_param,rho).to(DEVICE)
        self.assertEqual(fconv3d(dummy_input).shape[2], dummy_input.shape[2]//2)
        self.assertEqual(fconv3d(dummy_input).shape[3], dummy_input.shape[3]//2)

    # tests for FConv3dBLock
    def test_fconv3d_blcok_num_channels(self):
        """Test if the number of channels is the same"""
        k=3
        cin=32
        cout=64
        stride=1
        nu_param=2
        rho=6
        dummy_input = torch.randn(16, cin, 32, 32).to(DEVICE)
        fconv3dblock = FConv3dBlock(cin, cout, stride, k, nu_param, rho).to(DEVICE)
        self.assertEqual(fconv3dblock(dummy_input).shape[1], cout)

    def test_fconv3d_block_padding_test(self):
        """Test if the padding is the same"""
        k=3
        cin=32
        cout=64
        stride=1
        nu_param=2
        rho=6
        dummy_input = torch.randn(16, cin, 32, 32).to(DEVICE)
        fconv3dblock = FConv3dBlock(cin, cout, stride, k, nu_param, rho).to(DEVICE)
        self.assertEqual(fconv3dblock(dummy_input).shape[2], dummy_input.shape[2])
        self.assertEqual(fconv3dblock(dummy_input).shape[3], dummy_input.shape[3])

    def test_fconv3d_block_stride_test(self):
        """Test if the stride is the same"""
        k=3
        cin=32
        cout=64
        stride=2
        nu_param=2
        rho=6
        dummy_input = torch.randn(16, cin, 32, 32).to(DEVICE)
        fconv3dblock = FConv3dBlock(cin, cout, stride, k, nu_param, rho).to(DEVICE)
        self.assertEqual(fconv3dblock(dummy_input).shape[2], dummy_input.shape[2]//2)
        self.assertEqual(fconv3dblock(dummy_input).shape[3], dummy_input.shape[3]//2)

    # tests for FCNN
    def test_fcnn34_bs(self):
        """Test if the batch size is the same"""
        nu_param=2
        rho=6
        batch_size=16
        dummy_input = torch.randn(batch_size, 3, 32, 32).to(DEVICE)
        fcnn = fcnn34(nu_param, rho).to(DEVICE)
        out = fcnn(dummy_input)
        self.assertEqual(out.shape[0], batch_size)

    def test_fcnn34_classes(self):
        """Test if the number of classes is equal to output dimension of FCNN"""
        nu_param=2
        rho=6
        batch_size=16
        dummy_input = torch.randn(batch_size, 3, 32, 32).to(DEVICE)
        fcnn = fcnn34(nu_param, rho).to(DEVICE)
        out = fcnn(dummy_input)
        self.assertEqual(out.shape[1], NUM_CLASSES)

    # tests for ResNet (teacher model)
    def test_resnet34_bs(self):
        """Test if the batch size is the same"""
        batch_size=16
        dummy_input = torch.randn(batch_size, 3, 32, 32).to(DEVICE)
        resnet = resnet34().to(DEVICE)
        out = resnet(dummy_input)
        self.assertEqual(out.shape[0], batch_size)

    def test_resnet34_classes(self):
        """Test if the number of classes is equal to output dimension of ResNet"""
        batch_size=16
        dummy_input = torch.randn(batch_size, 3, 32, 32).to(DEVICE)
        resnet = resnet34().to(DEVICE)
        out = resnet(dummy_input)
        self.assertEqual(out.shape[1], NUM_CLASSES)

if __name__ == "__main__":
    NUM_CLASSES=100
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    unittest.main()
