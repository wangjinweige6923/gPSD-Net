#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet32Baseline(nn.Module):
    """
    Standard U-Net baseline used only by the new dual-source Step6 scripts.

    Design:
    - base_channels = 32
    - 3 encoder stages + 1 bottleneck
    - no deep supervision
    - no gPDC / LMM / SDPM
    - single sigmoid output
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 32):
        super().__init__()
        c = int(base_channels)

        self.enc1 = DoubleConv(in_channels, c)
        self.enc2 = DoubleConv(c, c * 2)
        self.enc3 = DoubleConv(c * 2, c * 4)
        self.bottleneck = DoubleConv(c * 4, c * 8)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.up1 = nn.ConvTranspose2d(c * 8, c * 4, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(c * 8, c * 4)

        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(c * 4, c * 2)

        self.up3 = nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(c * 2, c)

        self.final_conv = nn.Conv2d(c, out_channels, kernel_size=1)

    @staticmethod
    def _match_size(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if src.shape[2:] == ref.shape[2:]:
            return src
        return torch.nn.functional.interpolate(
            src,
            size=ref.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d1 = self.up1(b)
        d1 = self._match_size(d1, e3)
        d1 = self.dec1(torch.cat([d1, e3], dim=1))

        d2 = self.up2(d1)
        d2 = self._match_size(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d3 = self.up3(d2)
        d3 = self._match_size(d3, e1)
        d3 = self.dec3(torch.cat([d3, e1], dim=1))

        return torch.sigmoid(self.final_conv(d3))


def create_unet_baseline(base_channels: int = 32) -> UNet32Baseline:
    return UNet32Baseline(in_channels=1, out_channels=1, base_channels=base_channels)
