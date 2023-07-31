import torch
import torch.nn as nn

from .module import DoubleConv

__all__ = ['Decoder']
class Decoder(nn.Module):
    def __init__(self, out_channels, filters = [32, 64, 128, 256, 512]):
        super(Decoder, self).__init__()
        
        self.upsample1 = nn.ConvTranspose3d(filters[4], filters[4], kernel_size=2, stride=2)
        self.dconv_up1 = DoubleConv(filters[4] + filters[3], filters[3])
        
        self.upsample2 = nn.ConvTranspose3d(filters[3], filters[3], kernel_size=2, stride=2)
        self.dconv_up2 = DoubleConv(filters[3] + filters[2], filters[2])
        
        self.upsample4 = nn.ConvTranspose3d(filters[2], filters[2], kernel_size=2, stride=2)
        self.dconv_up3 = DoubleConv(filters[2] + filters[1], filters[1])
        
        self.upsample5 = nn.ConvTranspose3d(filters[1], filters[1], kernel_size=2, stride=2)
        self.dconv_up4 = DoubleConv(filters[1] + filters[0], filters[0])
        
        self.conv_last = nn.Conv3d(filters[0], out_channels, 1)
        
    def forward(self, x1, x2, x3, x4, x5):
        x = self.upsample1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.dconv_up1(x)
        
        x = self.upsample2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.dconv_up2(x)
        
        x = self.upsample4(x)
        x = torch.cat([x, x2], dim=1)
        x = self.dconv_up3(x)
        
        x = self.upsample5(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dconv_up4(x)
        
        out = self.conv_last(x)
        
        return out
        
    