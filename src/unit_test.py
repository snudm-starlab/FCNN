import torch
from torch import nn
import numpy as np
import unittest, argparse
from models.fcnn import FConv3d, FConv3dBlock, FCNN, fcnn34
from models.resnet import resnet34
from utils import get_flops 

NUM_CLASSES=100
class FCNNTest(unittest.TestCase):
    """ A class for unittest of FCNN """

    # tests for FConv3d
    def test_FConv3d_num_channels(self):
        k=3; cin=32; cout=64;s=1;
        nu=2;rho=6
        x = torch.randn(16, cin, 32, 32)
        fconv3d = FConv3d(k,cin,cout,s,nu,rho)
        conv2d = nn.Conv2d(cin,cout,k,stride=s, padding='same')
        self.assertEqual(fconv3d(x).shape[1], conv2d(x).shape[1])
        self.assertEqual(fconv3d(x).shape[1], cout)

    def test_FConv3d_padding_test(self):
        k=3; cin=32; cout=64;s=1
        nu=2;rho=6
        x = torch.randn(16, cin, 32, 32)
        fconv3d = FConv3d(k,cin,cout,s,nu,rho)
        conv2d = nn.Conv2d(cin,cout,k,s)
        self.assertEqual(fconv3d(x).shape[2], x.shape[2])
        self.assertEqual(fconv3d(x).shape[3], x.shape[3])

    def test_FConv3d_stride_test(self):
        k=3; cin=32; cout=64;s=2
        nu=2;rho=6
        x = torch.randn(16, cin, 32, 32)
        fconv3d = FConv3d(k,cin,cout,s,nu,rho)
        conv2d = nn.Conv2d(cin,cout,k,s)
        self.assertEqual(fconv3d(x).shape[2], x.shape[2]//2)
        self.assertEqual(fconv3d(x).shape[3], x.shape[3]//2)
    
    # tests for FConv3dBLock
    def test_FConv3dBlcok_num_channels(self):
        k=3; cin=32; cout=64;s=1
        nu=2;rho=6
        x = torch.randn(16, cin, 32, 32)
        fconv3dblock = FConv3dBlock(cin, cout, s, k, nu, rho)
        self.assertEqual(fconv3dblock(x).shape[1], cout)

    def test_FConv3dBlock_padding_test(self):
        k=3; cin=32; cout=64;s=1
        nu=2;rho=6
        x = torch.randn(16, cin, 32, 32)
        fconv3dblock = FConv3dBlock(cin, cout, s, k, nu, rho)
        self.assertEqual(fconv3dblock(x).shape[2], x.shape[2])
        self.assertEqual(fconv3dblock(x).shape[3], x.shape[3])

    def test_FConv3dBlock_stride_test(self):
        k=3; cin=32; cout=64;s=2
        nu=2;rho=6
        x = torch.randn(16, cin, 32, 32)
        fconv3dblock = FConv3dBlock(cin, cout, s, k, nu, rho)
        self.assertEqual(fconv3dblock(x).shape[2], x.shape[2]//2)
        self.assertEqual(fconv3dblock(x).shape[3], x.shape[3]//2)

    # tests for FCNN
    def test_FCNN34_bs(self):
        NU=2; RHO=6; bs=16
        x = torch.randn(bs, 3, 32, 32)
        FCNN34 = fcnn34(NU, RHO)
        out = FCNN34(x)
        self.assertEqual(out.shape[0], bs)

    def test_FCNN34_classes(self):
        NU=2; RHO=6; bs=16
        x = torch.randn(bs, 3, 32, 32)
        FCNN34 = fcnn34(NU, RHO)
        out = FCNN34(x)
        self.assertEqual(out.shape[1], NUM_CLASSES)

    # tests for ResNet (teacher model)
    def test_ResNet34_bs(self):
        bs=16
        x = torch.randn(bs, 3, 32, 32)
        ResNet34 = resnet34()
        out = ResNet34(x)
        self.assertEqual(out.shape[0], bs)

    def test_ResNet34_classes(self):
        bs=16
        x = torch.randn(bs, 3, 32, 32)
        ResNet34 = resnet34()
        out = ResNet34(x)
        self.assertEqual(out.shape[1], NUM_CLASSES)
