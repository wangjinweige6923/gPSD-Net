import torch
import torch.nn as nn
import torch.nn.functional as F
from pdc_convolutions import get_conv_layer
from improved_pdc_convolutions import get_improved_conv_layer, normalize_config_string, parse_config_string
from SDA import Spatial_Dependency_Perception_Module


class LightweightMultiscaleModuleV2(nn.Module):
    """
    Legacy LMM kept here only for checkpoint/eval compatibility.

    The standalone LMM.py file was removed so reviewers no longer see it as a
    separate module, but old checkpoints must still load with identical module
    names and parameter structure.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        use_multiscale: bool = True,
        use_residual: bool = True,
    ):
        super(LightweightMultiscaleModuleV2, self).__init__()

        if out_channels is None:
            out_channels = in_channels

        kernels = [1, 3, 5]

        self.use_multiscale = use_multiscale
        self.use_residual = use_residual

        if not use_multiscale:
            if in_channels != out_channels:
                self.identity = nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, bias=False
                )
            else:
                self.identity = nn.Identity()
        else:
            self.branches = nn.ModuleList()
            for kernel_size in kernels:
                padding = kernel_size // 2
                branch = nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        in_channels,
                        kernel_size=kernel_size,
                        padding=padding,
                        groups=in_channels,
                        bias=False,
                    ),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(inplace=True),
                )
                self.branches.append(branch)

            fusion_in_channels = in_channels * len(kernels)
            self.fusion = nn.Sequential(
                nn.Conv2d(fusion_in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

            if use_residual:
                if in_channels != out_channels:
                    self.residual_conv = nn.Sequential(
                        nn.Conv2d(
                            in_channels, out_channels, kernel_size=1, bias=False
                        ),
                        nn.BatchNorm2d(out_channels),
                    )
                else:
                    self.residual_conv = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_multiscale:
            return self.identity(x)

        identity = x
        branch_outputs = []
        for branch in self.branches:
            branch_outputs.append(branch(x))

        concat_features = torch.cat(branch_outputs, dim=1)
        out = self.fusion(concat_features)

        if self.use_residual:
            if hasattr(self, "residual_conv") and self.residual_conv is not None:
                identity = self.residual_conv(identity)
            out = out + identity

        return out


class PDCUNetBlock(nn.Module):
    """PDC-UNet基础块"""
    def __init__(self, in_channels, out_channels, conv_types=['V', 'V', 'V', 'V'], use_gpdc=False):
        super(PDCUNetBlock, self).__init__()
        self.conv_layers = nn.ModuleList()

        # 第一个卷积
        if use_gpdc:
            conv1 = get_improved_conv_layer(conv_types[0], in_channels, out_channels, use_gpdc=True)
        else:
            conv1 = get_conv_layer(conv_types[0], in_channels, out_channels)
        self.conv_layers.append(conv1)

        # 剩余三个卷积
        for i in range(1, 4):
            if use_gpdc:
                conv = get_improved_conv_layer(conv_types[i], out_channels, out_channels, use_gpdc=True)
            else:
                conv = get_conv_layer(conv_types[i], out_channels, out_channels)
            self.conv_layers.append(conv)

        # 批归一化和激活函数
        self.bn_layers = nn.ModuleList([nn.BatchNorm2d(out_channels) for _ in range(4)])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = conv(x)
            x = bn(x)
            x = self.relu(x)
        return x


class ImprovedPDCUNetBlock(nn.Module):
    """改进的PDC-UNet块，支持残差连接"""
    def __init__(self, in_channels, out_channels, conv_types=['V', 'V', 'V', 'V'],
                 use_gpdc=False, use_residual=False):
        super(ImprovedPDCUNetBlock, self).__init__()
        self.use_residual = use_residual

        self.conv_layers = nn.ModuleList()

        # 第一个卷积
        if use_gpdc:
            conv1 = get_improved_conv_layer(conv_types[0], in_channels, out_channels, use_gpdc=True)
        else:
            conv1 = get_conv_layer(conv_types[0], in_channels, out_channels)
        self.conv_layers.append(conv1)

        # 剩余三个卷积
        for i in range(1, 4):
            if use_gpdc:
                conv = get_improved_conv_layer(conv_types[i], out_channels, out_channels, use_gpdc=True)
            else:
                conv = get_conv_layer(conv_types[i], out_channels, out_channels)
            self.conv_layers.append(conv)

        # 批归一化和激活函数
        self.bn_layers = nn.ModuleList([nn.BatchNorm2d(out_channels) for _ in range(4)])
        self.relu = nn.ReLU(inplace=True)

        # 残差连接的通道匹配
        if use_residual and in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            self.residual_bn = nn.BatchNorm2d(out_channels)
        else:
            self.residual_conv = None

    def forward(self, x):
        identity = x
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = conv(x)
            x = bn(x)
            x = self.relu(x)

        if self.use_residual:
            if self.residual_conv is not None:
                identity = self.residual_conv(identity)
                identity = self.residual_bn(identity)
            x = x + identity
            x = self.relu(x)
        return x


class PDCUNet(nn.Module):
    """PDC-UNet完整架构 + LMM / SDPM 插件 + 可选多尺度深度监督"""
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 base_channels: int = 32,
                 config_str: str = 'baseline',
                 use_gpdc: bool = False,
                 use_residual: bool = False,
                 use_lmm: bool = False,
                 use_sdpm: bool = False,
                 use_deep_supervision: bool = False):
        super(PDCUNet, self).__init__()

        self.use_lmm = use_lmm
        self.use_sdpm = use_sdpm
        self.use_deep_supervision = use_deep_supervision

        # 解析配置字符串获取每个位置的卷积类型
        if config_str == 'baseline':
            conv_config = ['V'] * 12
        else:
            conv_config = parse_config_string(config_str)

        # 确保配置长度正确（3个编码器阶段，每个4个卷积）
        if len(conv_config) != 12:
            raise ValueError(f"Config must have 12 convolutions, got {len(conv_config)}")

        # -------------------------
        # 编码器：全 gPDC 可选 + 残差
        # -------------------------
        self.enc1 = ImprovedPDCUNetBlock(
            in_channels, base_channels,
            conv_config[0:4],
            use_gpdc=use_gpdc,
            use_residual=use_residual,
        )
        self.enc2 = ImprovedPDCUNetBlock(
            base_channels, base_channels * 2,
            conv_config[4:8],
            use_gpdc=use_gpdc,
            use_residual=use_residual,
        )
        self.enc3 = ImprovedPDCUNetBlock(
            base_channels * 2, base_channels * 4,
            conv_config[8:12],
            use_gpdc=use_gpdc,
            use_residual=use_residual,
        )

        # 瓶颈层：标准卷积（不再用 gPDC）
        self.bottleneck = ImprovedPDCUNetBlock(
            base_channels * 4, base_channels * 8,
            ['V', 'V', 'V', 'V'],
            use_gpdc=False,
            use_residual=use_residual,
        )

        # -------------------------
        # 解码器（不使用 gPDC，通道布局固定）
        # -------------------------
        # dec1 输入 = up1(8C) + e3(4C) = 12C
        self.dec1 = ImprovedPDCUNetBlock(
            base_channels * 8 + base_channels * 4,
            base_channels * 4,
            ['V', 'V', 'V', 'V'],
            use_gpdc=False,
            use_residual=use_residual,
        )
        # dec2 输入 = up2(4C) + e2(2C) = 6C
        self.dec2 = ImprovedPDCUNetBlock(
            base_channels * 4 + base_channels * 2,
            base_channels * 2,
            ['V', 'V', 'V', 'V'],
            use_gpdc=False,
            use_residual=use_residual,
        )
        # dec3 输入 = up3(2C) + e1(C) = 3C
        self.dec3 = ImprovedPDCUNetBlock(
            base_channels * 2 + base_channels,
            base_channels,
            ['V', 'V', 'V', 'V'],
            use_gpdc=False,
            use_residual=use_residual,
        )

        # 下采样和上采样层
        self.pool = nn.MaxPool2d(2)
        self.up1 = nn.ConvTranspose2d(base_channels * 8, base_channels * 8, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 4, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(base_channels * 2, base_channels * 2, kernel_size=2, stride=2)

        # -------------------------
        # LMM：作用在解码输出
        # -------------------------
        if self.use_lmm:
            self.lmm1 = LightweightMultiscaleModuleV2(
                in_channels=base_channels * 4,
                out_channels=base_channels * 4,
                use_multiscale=True,
                use_residual=True,
            )
            self.lmm2 = LightweightMultiscaleModuleV2(
                in_channels=base_channels * 2,
                out_channels=base_channels * 2,
                use_multiscale=True,
                use_residual=True,
            )
            self.lmm3 = LightweightMultiscaleModuleV2(
                in_channels=base_channels,
                out_channels=base_channels,
                use_multiscale=True,
                use_residual=True,
            )

        # -------------------------
        # SDPM：Q = decoder 上采样语义特征，K/V = 升通道后的 skip 细节特征
        # 作用在 up 与 skip 之间，保证 dec1/2/3 的输入通道仍是 12C / 6C / 3C
        # -------------------------
        if self.use_sdpm:
            # 1/4 分辨率：up1 通道 = 8C，skip(e3) 通道 = 4C → 将 e3 升到 8C
            self.sdpm1_high_proj = nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=1, bias=False)
            self.sdpm1 = Spatial_Dependency_Perception_Module(
                dim=base_channels * 8,
                patch=4,
                inter_dim=base_channels * 8,
            )
            # 1/2 分辨率：up2 通道 = 4C，skip(e2) 通道 = 2C → 将 e2 升到 4C
            self.sdpm2_high_proj = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=1, bias=False)
            self.sdpm2 = Spatial_Dependency_Perception_Module(
                dim=base_channels * 4,
                patch=4,
                inter_dim=base_channels * 4,
            )
            # 1/1 分辨率：up3 通道 = 2C，skip(e1) 通道 = C → 将 e1 升到 2C
            self.sdpm3_high_proj = nn.Conv2d(base_channels, base_channels * 2, kernel_size=1, bias=False)
            self.sdpm3 = Spatial_Dependency_Perception_Module(
                dim=base_channels * 2,
                patch=4,
                inter_dim=base_channels * 2,
            )

        # -------------------------
        # 最终输出层（1/1 主输出）+ 可选侧输出头
        # -------------------------
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

        # 深度监督侧输出头：1/2、1/4 分辨率
        # 注意：只有在 use_deep_supervision=True 时才创建，确保 Step5 旧模型结构完全不变
        if self.use_deep_supervision:
            # dec2: 1/2, 通道数 = 2C
            self.ds_1_2 = nn.Conv2d(base_channels * 2, out_channels, kernel_size=1)
            # dec1: 1/4, 通道数 = 4C
            self.ds_1_4 = nn.Conv2d(base_channels * 4, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        # 编码器路径
        e1 = self.enc1(x)                 # 1/1,   C
        e2 = self.enc2(self.pool(e1))     # 1/2,  2C
        e3 = self.enc3(self.pool(e2))     # 1/4,  4C

        # 瓶颈层：1/8, 8C
        b = self.bottleneck(self.pool(e3))

        # -----------------------
        # 解码器 1/4 层
        # -----------------------
        d1_up = self.up1(b)               # 1/4, 8C
        if self.use_sdpm:
            e3_high = self.sdpm1_high_proj(e3)          # 4C -> 8C
            d1_refined = self.sdpm1(d1_up, e3_high)     # Q = d1_up, K/V = e3_high
            d1_in = torch.cat([d1_refined, e3], dim=1)  # 8C + 4C = 12C
        else:
            d1_in = torch.cat([d1_up, e3], dim=1)
        d1 = self.dec1(d1_in)             # 1/4, 4C
        if self.use_lmm:
            d1 = self.lmm1(d1)

        # -----------------------
        # 解码器 1/2 层
        # -----------------------
        d2_up = self.up2(d1)              # 1/2, 4C
        if self.use_sdpm:
            e2_high = self.sdpm2_high_proj(e2)          # 2C -> 4C
            d2_refined = self.sdpm2(d2_up, e2_high)
            d2_in = torch.cat([d2_refined, e2], dim=1)  # 4C + 2C = 6C
        else:
            d2_in = torch.cat([d2_up, e2], dim=1)
        d2 = self.dec2(d2_in)             # 1/2, 2C
        if self.use_lmm:
            d2 = self.lmm2(d2)

        # -----------------------
        # 解码器 1/1 层
        # -----------------------
        d3_up = self.up3(d2)              # 1/1, 2C
        if self.use_sdpm:
            e1_high = self.sdpm3_high_proj(e1)          # C -> 2C
            d3_refined = self.sdpm3(d3_up, e1_high)
            d3_in = torch.cat([d3_refined, e1], dim=1)  # 2C + C = 3C
        else:
            d3_in = torch.cat([d3_up, e1], dim=1)
        d3 = self.dec3(d3_in)             # 1/1, C
        if self.use_lmm:
            d3 = self.lmm3(d3)

        # 主输出 1/1
        p0 = self.final_conv(d3)

        # 兼容旧版本：未开启深度监督时，只返回单一输出
        if not self.use_deep_supervision:
            return torch.sigmoid(p0)

        # 开启深度监督：增加 1/2、1/4 侧输出
        p1 = self.ds_1_2(d2)   # 1/2
        p2 = self.ds_1_4(d1)   # 1/4

        # 返回三尺度概率图：p0 (1/1), p1 (1/2), p2 (1/4)
        return torch.sigmoid(p0), torch.sigmoid(p1), torch.sigmoid(p2)


def create_model(config_str: str = 'baseline',
                 channels: int = 32,
                 use_gpdc: bool = False,
                 use_residual: bool = False,
                 use_lmm: bool = False,
                 use_sdpm: bool = False,
                 use_deep_supervision: bool = False):
    """创建模型的便利函数"""
    return PDCUNet(
        in_channels=1,
        out_channels=1,
        base_channels=channels,
        config_str=config_str,
        use_gpdc=use_gpdc,
        use_residual=use_residual,
        use_lmm=use_lmm,
        use_sdpm=use_sdpm,
        use_deep_supervision=use_deep_supervision,
    )


# 第一步实验的14个配置
STEP1_CONFIGS = [
    'baseline',
    '[CARV]×3',
    'C-[V]×11',
    'A-[V]×11',
    'R-[V]×11',
    '[CVVV]×3',
    '[AVVV]×3',
    '[RVVV]×3',
    '[CCCV]×3',
    '[AAAV]×3',
    '[RRRV]×3',
    '[C]×12',
    '[A]×12',
    '[R]×12',
]

# 第二步实验的配置和通道数组合
STEP2_CONFIGS = ['baseline', 'C-[V]×11']
STEP2_CHANNELS = [16, 32, 64]

# Prefer ASCII config tokens in new code paths while keeping legacy aliases
# accepted by normalize_config_string / parse_config_string.
STEP1_CONFIGS = [
    'baseline',
    '[CARV]x3',
    'C-[V]x11',
    'A-[V]x11',
    'R-[V]x11',
    '[CVVV]x3',
    '[AVVV]x3',
    '[RVVV]x3',
    '[CCCV]x3',
    '[AAAV]x3',
    '[RRRV]x3',
    '[C]x12',
    '[A]x12',
    '[R]x12',
]
STEP2_CONFIGS = ['baseline', 'C-[V]x11']

# 第三步实验的改进技术配置
STEP3_IMPROVEMENTS = {
    'baseline': {'use_residual': False, 'use_gpdc': False},
    'residual_only': {'use_residual': True, 'use_gpdc': False},
    'gpdc_only': {'use_residual': False, 'use_gpdc': True},
    'full_improvement': {'use_residual': True, 'use_gpdc': True},
}

def get_model_name(config_str, channels, use_gpdc=False, use_residual=False):
    """生成模型名称"""
    name_parts = [config_str.replace('×', 'x'), f'C{channels}']
    if use_gpdc:
        name_parts.append('gPDC')
    if use_residual:
        name_parts.append('Residual')
    return '_'.join(name_parts)


def get_model_name(config_str, channels, use_gpdc=False, use_residual=False):
    """Generate a stable model name regardless of legacy config encoding."""
    try:
        normalized_config = normalize_config_string(config_str)
    except ValueError:
        normalized_config = str(config_str).strip()

    name_parts = [normalized_config, f'C{channels}']
    if use_gpdc:
        name_parts.append('gPDC')
    if use_residual:
        name_parts.append('Residual')
    return '_'.join(name_parts)
