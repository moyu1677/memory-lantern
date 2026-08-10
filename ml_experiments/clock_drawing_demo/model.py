"""不依赖 torchvision 的轻量画钟评分网络。"""

from __future__ import annotations

import torch
from torch import nn


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class DepthwiseSeparable(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, inputs):
        return self.block(inputs)


class ClockScoreNet(nn.Module):
    """保留粗粒度空间布局的轻量六级画钟基线模型。"""

    def __init__(self, num_classes=6, dropout=0.25):
        super().__init__()
        self.features = nn.Sequential(
            ConvNormAct(1, 24, stride=2),
            DepthwiseSeparable(24, 40, stride=2),
            DepthwiseSeparable(40, 64, stride=2),
            DepthwiseSeparable(64, 96, stride=2),
            DepthwiseSeparable(96, 144, stride=2),
            DepthwiseSeparable(144, 192, stride=2),
            # 画钟评分依赖数字与指针的相对位置，不能像普通物体分类那样
            # 直接压成单个全局向量，因此保留4×4空间网格。
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(192 * 4 * 4, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs):
        return self.classifier(self.features(inputs))


def model_summary(model):
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return {"parameters": total, "trainable_parameters": trainable}
