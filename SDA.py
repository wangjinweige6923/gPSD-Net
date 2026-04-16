import torch
import numpy as np
import torch.nn as nn
from einops import rearrange


def autopad(k, p=None, d=1):  # kernel, padding, dilation
   
    if d > 1:
        
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # 实际卷积核大小
    
    if p is None:
               p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # 自动填充
    return p

class Conv(nn.Module):
   
    default_act = nn.SiLU()  

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
       
        super().__init__()  
        
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
       
        self.bn = nn.BatchNorm2d(c2)
        
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
     
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
    
        return self.act(self.conv(x))

class Spatial_Dependency_Perception_Module(nn.Module):
  
    def __init__(self,
                 dim=256, 
                 patch=None, 
                 inter_dim=None 
                 ):
        super(Spatial_Dependency_Perception_Module, self).__init__()
        self.dim = dim 
        self.inter_dim = inter_dim  
    
        if self.inter_dim == None:
            self.inter_dim = dim
      
        self.conv_q = nn.Sequential(*[nn.Conv2d(dim, self.inter_dim, 1, padding=0, bias=False),
                                      nn.GroupNorm(32, self.inter_dim)])
        
        self.conv_k = nn.Sequential(*[nn.Conv2d(dim, self.inter_dim, 1, padding=0, bias=False),
                                      nn.GroupNorm(32, self.inter_dim)])

        self.softmax = nn.Softmax(dim=-1)
  
        self.patch_size = (patch, patch)
        
        self.conv1x1 = Conv(self.dim, self.inter_dim, 1)

    def forward(self, x_low, x_high):
     
        b_, _, h_, w_ = x_low.size()

    
        q = rearrange(self.conv_q(x_low), 'b c (h p1) (w p2) -> (b h w) c (p1 p2)',
                      p1=self.patch_size[0], p2=self.patch_size[1])
   
        q = q.transpose(1, 2) 


        k = rearrange(self.conv_k(x_high), 'b c (h p1) (w p2) -> (b h w) c (p1 p2)',
                      p1=self.patch_size[0], p2=self.patch_size[1])


        attn = torch.matmul(q, k)
    
        attn = attn / np.power(self.inter_dim, 0.5)
  
        attn = self.softmax(attn)

      
        v = k.transpose(1, 2)  

      
        output = torch.matmul(attn, v) 

       
        output = rearrange(output.transpose(1, 2).contiguous(),
                           '(b h w) c (p1 p2) -> b c (h p1) (w p2)',
                           p1=self.patch_size[0], p2=self.patch_size[1],
                           h=h_ // self.patch_size[0], w=w_ // self.patch_size[1])

     
        if self.dim != self.inter_dim:
            x_low = self.conv1x1(x_low)

     
        return output + x_low

if __name__ == '__main__':
  
    input1 = torch.randn(1, 64, 128, 128)
   
    input2 = torch.randn(1, 64, 128, 128)

    sdp = Spatial_Dependency_Perception_Module(64, 8, 64)
    output = sdp(input1, input2)
    print(f"输入张量1形状: {input1.shape}")
    print(f"输入张量2形状: {input2.shape}")
    print(f"输出张量形状: {output.shape}")
