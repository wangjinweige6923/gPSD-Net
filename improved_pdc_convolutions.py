import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re
from pdc_convolutions import CDConv, ADConv, RDConv, VanillaConv

class ImprovedCDConv(nn.Module):
    """改进的Center Difference Convolution with gPDC双通道"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(ImprovedCDConv, self).__init__()
        
        
        self.gradient_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        
    
        self.intensity_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        
        
        kernel = torch.zeros(kernel_size, kernel_size)
        center = kernel_size // 2
        kernel[center, center] = 4  
        kernel[center-1, center] = -1  
        kernel[center+1, center] = -1  
        kernel[center, center-1] = -1  
        kernel[center, center+1] = -1 
        
        
        self.register_buffer('cd_kernel', kernel.unsqueeze(0).unsqueeze(0))
        
        
        self.alpha = nn.Parameter(torch.tensor(0.7))  # 梯度通道权重较大
        
    def forward(self, x):
        
        cd_features = F.conv2d(x, self.cd_kernel.repeat(x.shape[1], 1, 1, 1), 
                              groups=x.shape[1], padding=1)
        gradient_out = self.gradient_conv(cd_features)
        
        
        intensity_out = self.intensity_conv(x)
        

        fused_output = self.alpha * gradient_out + (1 - self.alpha) * intensity_out
        
        return fused_output

class ImprovedADConv(nn.Module):
    """改进的Angular Difference Convolution with gPDC双通道"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(ImprovedADConv, self).__init__()
        
        self.gradient_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.intensity_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        
        
        kernel = torch.zeros(kernel_size, kernel_size)
       
        kernel[0, 0] = 0.5; kernel[2, 2] = 0.5   
        kernel[0, 2] = -0.5; kernel[2, 0] = -0.5  
        
        self.register_buffer('ad_kernel', kernel.unsqueeze(0).unsqueeze(0))
        self.alpha = nn.Parameter(torch.tensor(0.7))  
        
    def forward(self, x):
        
        ad_features = F.conv2d(x, self.ad_kernel.repeat(x.shape[1], 1, 1, 1), 
                              groups=x.shape[1], padding=1)
        gradient_out = self.gradient_conv(ad_features)
        
        
        intensity_out = self.intensity_conv(x)
        
        
        fused_output = self.alpha * gradient_out + (1 - self.alpha) * intensity_out
        
        return fused_output

class ImprovedRDConv(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(ImprovedRDConv, self).__init__()
        
        self.gradient_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.intensity_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        
        
        kernel = torch.zeros(kernel_size, kernel_size)
        center = kernel_size // 2
        kernel[center, center] = 2  
        
        if center-1 >= 0:
            kernel[center-1, center] = -0.5
        if center+1 < kernel_size:
            kernel[center+1, center] = -0.5
        if center-1 >= 0:
            kernel[center, center-1] = -0.5
        if center+1 < kernel_size:
            kernel[center, center+1] = -0.5
        
        self.register_buffer('rd_kernel', kernel.unsqueeze(0).unsqueeze(0))
        self.alpha = nn.Parameter(torch.tensor(0.7))  # 同样偏向梯度通道
        
    def forward(self, x):
        
        rd_features = F.conv2d(x, self.rd_kernel.repeat(x.shape[1], 1, 1, 1), 
                              groups=x.shape[1], padding=1)
        gradient_out = self.gradient_conv(rd_features)
        
       
        intensity_out = self.intensity_conv(x)
        
        
        fused_output = self.alpha * gradient_out + (1 - self.alpha) * intensity_out
        
        return fused_output

class LearnableAlphaFusion(nn.Module):
    
    def __init__(self, channels):
        super(LearnableAlphaFusion, self).__init__()
       
        self.alpha = nn.Parameter(torch.ones(channels) * 0.7)
        
    def forward(self, gradient_features, intensity_features):
        """
        Args:
            gradient_features: 梯度通道特征 [B, C, H, W]
            intensity_features: 强度通道特征 [B, C, H, W]
        """
        
        alpha_expanded = self.alpha.view(1, -1, 1, 1)
        
       
        fused = alpha_expanded * gradient_features + (1 - alpha_expanded) * intensity_features
        
        return fused

def get_improved_conv_layer(conv_type, in_channels, out_channels, use_gpdc=True, **kwargs):
    
    if use_gpdc:
        conv_dict = {
            'C': ImprovedCDConv,
            'A': ImprovedADConv,
            'R': ImprovedRDConv,
            'V': VanillaConv  
        }
    else:
      
        conv_dict = {
            'C': CDConv,
            'A': ADConv,
            'R': RDConv,
            'V': VanillaConv
        }
    
    if conv_type not in conv_dict:
        raise ValueError(f"Unknown conv type: {conv_type}")
    
    return conv_dict[conv_type](in_channels, out_channels, **kwargs)


def _legacy_parse_config_string(config_str):
    """Parse legacy config strings that may contain mojibake separators.
    
    Examples:
        'C-[V]×11' -> ['C'] + ['V'] * 11
        '[CARV]×3' -> ['C', 'A', 'R', 'V'] * 3
        '[CVVV]×3' -> ['C', 'V', 'V', 'V'] * 3
    """
    if 'baseline' in config_str:
        return ['V'] * 12
    
    if '×' in config_str:
        if config_str.startswith('[') and ']×' in config_str:
            # 格式: [CARV]×3
            pattern, count = config_str.split('×')
            pattern = pattern.strip('[]')
            count = int(count)
            return list(pattern) * count
        else:
            # 格式: C-[V]×11
            parts = config_str.split('-')
            first_conv = parts[0]
            remaining_part = parts[1]  # [V]×11
            remaining_pattern, remaining_count = remaining_part.split('×')
            remaining_pattern = remaining_pattern.strip('[]')
            remaining_count = int(remaining_count)
            
            return [first_conv] + [remaining_pattern] * remaining_count
    
    # 格式: [C]×12
    if config_str.startswith('[') and config_str.endswith(']×12'):
        conv_type = config_str[1]  # 提取中间的字符
        return [conv_type] * 12
    
    raise ValueError(f"Unknown config format: {config_str}")


_CONFIG_REPEAT_RE = re.compile(r"^\[([CARV]+)\]([^0-9]*)?(\d+)$", re.IGNORECASE)
_CONFIG_PREFIX_RE = re.compile(r"^([CARV])-\[([CARV]+)\]([^0-9]*)?(\d+)$", re.IGNORECASE)


def normalize_config_string(config_str):
    """Normalize config strings across legacy mojibake and ASCII forms."""
    if config_str is None:
        raise ValueError("Config string is required.")

    config_text = str(config_str).strip().replace(" ", "")
    if not config_text:
        raise ValueError("Config string is empty.")

    if "baseline" in config_text.lower():
        return "baseline"

    match = _CONFIG_PREFIX_RE.fullmatch(config_text)
    if match:
        first_conv, remaining_pattern, _separator, repeat_count = match.groups()
        return f"{first_conv.upper()}-[{remaining_pattern.upper()}]x{int(repeat_count)}"

    match = _CONFIG_REPEAT_RE.fullmatch(config_text)
    if match:
        pattern, _separator, repeat_count = match.groups()
        return f"[{pattern.upper()}]x{int(repeat_count)}"

    raise ValueError(f"Unknown config format: {config_str}")


def parse_config_string(config_str):
    """Parse a config string into the 12 convolution type slots."""
    normalized = normalize_config_string(config_str)
    if normalized == "baseline":
        return ["V"] * 12

    match = _CONFIG_PREFIX_RE.fullmatch(normalized)
    if match:
        first_conv, remaining_pattern, _separator, repeat_count = match.groups()
        return [first_conv.upper()] + [remaining_pattern.upper()] * int(repeat_count)

    match = _CONFIG_REPEAT_RE.fullmatch(normalized)
    if match:
        pattern, _separator, repeat_count = match.groups()
        return list(pattern.upper()) * int(repeat_count)

    raise ValueError(f"Unknown config format: {config_str}")
