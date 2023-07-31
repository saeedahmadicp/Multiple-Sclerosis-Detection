import torch
import torch.nn as nn

from module import DoubleConv

__all__ = ['Encoder']
class Encoder(nn.Module):
    def __init__(self, in_channels, filters = [32, 64, 128, 256, 512]):
        super(Encoder, self).__init__() 
        
        self.dconv_down1 = DoubleConv(in_channels, filters[0], stride=1)
        self.dconv_down2 = DoubleConv(filters[0], filters[1], stride=2)
        self.dconv_down3 = DoubleConv(filters[1], filters[2], stride=2)
        self.dconv_down4 = DoubleConv(filters[2], filters[3], stride=2)
        self.dconv_down5 = DoubleConv(filters[3], filters[4], stride=2)
        
    def forward(self, x):
        x1 = self.dconv_down1(x) # 32
        x2 = self.dconv_down2(x1) # 64
        x3 = self.dconv_down3(x2) # 128
        x4 = self.dconv_down4(x3) # 256
        x5 = self.dconv_down5(x4) # 512
        
        return x1, x2, x3, x4, x5
        
        