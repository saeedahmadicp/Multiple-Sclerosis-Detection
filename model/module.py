import torch
import torch.nn as nn

__all__ = ['DoubleConv']

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DoubleConv, self).__init__()
        
        self.conv = nn.Sequential(
            ## 1st conv layer
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.GroupNorm(num_groups=out_channels//8, num_channels=out_channels),
            nn.LeakyReLU(inplace=True, negative_slope=0.01),
            
            ## 2nd conv layer
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=out_channels//8, num_channels=out_channels),
            nn.LeakyReLU(inplace=True, negative_slope=0.01),
        )
        
    def forward(self, x):
        return self.conv(x)