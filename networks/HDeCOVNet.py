# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 16:24:18 2024

@author: fares.bougourzi
"""
import torch
import torch.nn as nn

class GlobalMaxPool3d(nn.Module):
    """
    Reduce max over last three dimensions.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.max(dim=-1, keepdim=True)[0]
        x = x.max(dim=-2, keepdim=True)[0]
        x = x.max(dim=-3, keepdim=True)[0]
        return x
    
class ResNetBasicStem(nn.Module):
    def __init__(self, in_dim = 3):
        super(ResNetBasicStem, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv3d(in_dim, 16, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=[1, 3, 3], stride=[1, 2, 2], padding=[0, 1, 1], dilation=1, ceil_mode=False)
            )
    def forward(self, x):
        x= self.layer(x)
        
        return x
    
class BottleneckTransform(nn.Module):
    def __init__(self, in_dim, int_dim, out_dim, stride):
        super(BottleneckTransform, self).__init__()
        self.skip =nn.Sequential(nn.Conv3d(in_dim, out_dim, kernel_size=(1, 1, 1), stride=(1, stride, stride), bias=False),
                      nn.BatchNorm3d(out_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))

        self.a = nn.Sequential(
            nn.Conv3d(in_dim, int_dim, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(int_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
            )
        self.b = nn.Sequential(
            nn.Conv3d(int_dim, int_dim, kernel_size=(1, 3, 3), stride=(1, stride, stride), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(int_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
            )
        self.c = nn.Sequential(
            nn.Conv3d(int_dim, out_dim, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False),
            nn.BatchNorm3d(out_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )       
        self.act =  nn.ReLU(inplace=True)       
    def forward(self, x):
        skip = self.skip(x)
        x = self.a(x)
        x = self.b(x)
        x = self.c(x)
        x = self.act(x+skip)
        return x
       
class HDeCOVNet(nn.Module):
    def __init__(self, in_ch = 3):
        super(HDeCOVNet, self).__init__()
        self.stem = ResNetBasicStem(in_dim= in_ch)
        self.s1 = BottleneckTransform(in_dim = 16, int_dim=16, out_dim = 64, stride=1)
        self.max = nn.MaxPool3d(kernel_size=[2, 1, 1], stride=[2, 1, 1], padding=[0, 0, 0], dilation=1, ceil_mode=False)       
        self.s2 = BottleneckTransform(in_dim = 64, int_dim=32, out_dim = 128, stride=2)
        self.s3 = BottleneckTransform(in_dim = 128, int_dim=64, out_dim = 256, stride=1)
        self.s4 = BottleneckTransform(in_dim = 256, int_dim=128, out_dim = 512, stride=2)
        # self.max2 = nn.MaxPool3d(kernel_size=[1, 2, 2], stride=[1, 2, 2], padding=[0, 0, 0], dilation=1, ceil_mode=False)
        self.head = nn.Sequential(nn.AdaptiveMaxPool3d((16,14,14)),
                                  nn.Conv3d(512, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
                                  nn.BatchNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),      
                                  nn.ReLU(inplace=True),
                                  nn.MaxPool3d(kernel_size=[4, 2, 2], stride=[4, 2, 2], padding=[0, 0, 0], dilation=1, ceil_mode=False),
                                  nn.Conv3d(256, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
                                  nn.BatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                  nn.ReLU(inplace=True),
                                  nn.Conv3d(128,128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
                                  nn.BatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                  nn.ReLU(inplace=True),  
                                  GlobalMaxPool3d()
                                  # 
                                  )
        self.cls = nn.Linear(128, 2, bias=True)
        
    def forward(self, x):
        x = self.stem(x)
        x = self.s1(x)
        x = self.max(x)
        x = self.s2(x)
        
        x = self.s3(x)
        x = self.s4(x)
        x = self.head(x)
        x = x.view(x.shape[0], x.shape[1])  
        return self.cls(x)
####################################