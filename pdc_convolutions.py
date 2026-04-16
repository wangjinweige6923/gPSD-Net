import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CDConv(nn.Module):
    """Center Difference Convolution (CPDC)"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(CDConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        
        # 创建中心差分卷积核
        kernel = torch.zeros(kernel_size, kernel_size)
        kernel[kernel_size//2, kernel_size//2] = 8  # 中心
        kernel[0, 1] = -1; kernel[1, 0] = -1; kernel[1, 2] = -1; kernel[2, 1] = -1  # 上下左右
        kernel[0, 0] = -1; kernel[0, 2] = -1; kernel[2, 0] = -1; kernel[2, 2] = -1  # 四个角
        
        self.register_buffer('cd_kernel', kernel.unsqueeze(0).unsqueeze(0))
        
    def forward(self, x):
        # 应用中心差分卷积
        cd_features = F.conv2d(x, self.cd_kernel.repeat(x.shape[1], 1, 1, 1), 
                              groups=x.shape[1], padding=1)
        return self.conv(cd_features)

class ADConv(nn.Module):
    """Angular Difference Convolution (APDC)"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(ADConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        
        # 创建角度差分卷积核
        kernel = torch.zeros(kernel_size, kernel_size)
        kernel[0, 0] = 1; kernel[0, 2] = -1  # 左上到右上对角线
        kernel[2, 0] = -1; kernel[2, 2] = 1  # 左下到右下对角线
        
        self.register_buffer('ad_kernel', kernel.unsqueeze(0).unsqueeze(0))
        
    def forward(self, x):
        # 应用角度差分卷积
        ad_features = F.conv2d(x, self.ad_kernel.repeat(x.shape[1], 1, 1, 1), 
                              groups=x.shape[1], padding=1)
        return self.conv(ad_features)

class RDConv(nn.Module):
    """Radial Difference Convolution (RPDC)"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(RDConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        
        # 创建径向差分卷积核
        kernel = torch.zeros(kernel_size, kernel_size)
        center = kernel_size // 2
        for i in range(kernel_size):
            for j in range(kernel_size):
                if i == center and j == center:
                    kernel[i, j] = 4
                elif abs(i - center) + abs(j - center) == 1:  # 十字形
                    kernel[i, j] = -1
        
        self.register_buffer('rd_kernel', kernel.unsqueeze(0).unsqueeze(0))
        
    def forward(self, x):
        # 应用径向差分卷积
        rd_features = F.conv2d(x, self.rd_kernel.repeat(x.shape[1], 1, 1, 1), 
                              groups=x.shape[1], padding=1)
        return self.conv(rd_features)

class VanillaConv(nn.Module):
    """标准卷积"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(VanillaConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        
    def forward(self, x):
        return self.conv(x)

def get_conv_layer(conv_type, in_channels, out_channels, **kwargs):
    """根据类型获取对应的卷积层"""
    conv_dict = {
        'C': CDConv,
        'A': ADConv, 
        'R': RDConv,
        'V': VanillaConv
    }
    
    if conv_type not in conv_dict:
        raise ValueError(f"Unknown conv type: {conv_type}")
    
    return conv_dict[conv_type](in_channels, out_channels, **kwargs)