"""
CNN 深度学习模型
用于语音情感识别的卷积神经网络
支持：标准 CNN、残差连接、注意力机制
输入：Mel 频谱图特征 (n_mels, time_steps) 或原始特征向量
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union


class EmotionCNN(nn.Module):
    """
    语音情感识别 CNN 模型
    基于 Mel 频谱图的 2D 卷积神经网络
    """

    def __init__(
        self,
        n_mels: int = 128,
        time_steps: int = 94,  # 3秒 @ 16kHz, hop=512 -> ~94帧
        n_classes: int = 7,
        dropout_rate: float = 0.3,
    ):
        """
        初始化 CNN 模型

        Args:
            n_mels: Mel 滤波器数量（频谱图高度）
            time_steps: 时间步数（频谱图宽度）
            n_classes: 情绪类别数
            dropout_rate: Dropout 比率
        """
        super(EmotionCNN, self).__init__()

        self.n_mels = n_mels
        self.time_steps = time_steps
        self.n_classes = n_classes

        # 卷积块 1: 提取低频特征
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64x47 -> 32x23
        )

        # 卷积块 2: 提取中层特征
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x23 -> 16x11
        )

        # 卷积块 3: 提取高层特征
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x11 -> 8x5
        )

        # 卷积块 4: 深层特征
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # 全局平均池化 -> 256x1x1
        )

        # 计算全连接层输入维度
        self._feature_dim = 256

        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self._feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(128, n_classes),
        )

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量 (batch_size, 1, n_mels, time_steps)

        Returns:
            输出张量 (batch_size, n_classes)
        """
        # 卷积层
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # 展平
        x = x.view(x.size(0), -1)

        # 分类器
        x = self.classifier(x)
        return x


class EmotionResNet(nn.Module):
    """
    带残差连接的语音情感识别网络（ResNet风格）
    使用残差块解决深层网络的梯度消失问题
    """

    class ResidualBlock(nn.Module):
        """残差块"""

        def __init__(self, in_channels, out_channels, stride=1):
            super(EmotionResNet.ResidualBlock, self).__init__()

            self.conv1 = nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, stride=stride, padding=1, bias=False
            )
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(
                out_channels, out_channels,
                kernel_size=3, stride=1, padding=1, bias=False
            )
            self.bn2 = nn.BatchNorm2d(out_channels)

            # 捷径连接
            self.shortcut = nn.Sequential()
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_channels, out_channels,
                        kernel_size=1, stride=stride, bias=False
                    ),
                    nn.BatchNorm2d(out_channels),
                )

        def forward(self, x):
            residual = self.shortcut(x)
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += residual
            out = F.relu(out)
            return out

    def __init__(
        self,
        n_mels: int = 128,
        time_steps: int = 94,
        n_classes: int = 7,
        dropout_rate: float = 0.3,
    ):
        super(EmotionResNet, self).__init__()

        self.n_mels = n_mels
        self.time_steps = time_steps
        self.n_classes = n_classes

        # 初始卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # 残差层
        self.layer1 = self._make_layer(32, 64, num_blocks=2, stride=2)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)

        # 全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(128, n_classes),
        )

        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        """构建残差层"""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(self.ResidualBlock(in_channels, out_channels, stride))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class EmotionMLP(nn.Module):
    """
    多层感知机模型（基于122维特征向量）
    用于与传统 ML 模型进行对比的简单深度基线
    """

    def __init__(
        self,
        input_dim: int = 122,
        n_classes: int = 7,
        hidden_dims: List[int] = [256, 128, 64],
        dropout_rate: float = 0.3,
    ):
        """
        初始化 MLP 模型

        Args:
            input_dim: 输入特征维度
            n_classes: 情绪类别数
            hidden_dims: 隐藏层维度列表
            dropout_rate: Dropout 比率
        """
        super(EmotionMLP, self).__init__()

        self.input_dim = input_dim
        self.n_classes = n_classes
        self.hidden_dims = hidden_dims

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, n_classes))

        self.network = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def create_cnn_model(
    model_type: str = "cnn",
    n_mels: int = 128,
    time_steps: int = 94,
    n_classes: int = 7,
    input_dim: int = 122,
    **kwargs,
) -> nn.Module:
    """
    创建 CNN 模型的工厂函数

    Args:
        model_type: 模型类型 ('cnn', 'resnet', 'mlp')
        n_mels: Mel 频谱图高度
        time_steps: 时间步数
        n_classes: 类别数
        input_dim: MLP 输入维度
        **kwargs: 其他参数

    Returns:
        PyTorch 模型实例
    """
    if model_type == "cnn":
        return EmotionCNN(
            n_mels=n_mels,
            time_steps=time_steps,
            n_classes=n_classes,
            dropout_rate=kwargs.get("dropout_rate", 0.3),
        )
    elif model_type == "resnet":
        return EmotionResNet(
            n_mels=n_mels,
            time_steps=time_steps,
            n_classes=n_classes,
            dropout_rate=kwargs.get("dropout_rate", 0.3),
        )
    elif model_type == "mlp":
        return EmotionMLP(
            input_dim=input_dim,
            n_classes=n_classes,
            hidden_dims=kwargs.get("hidden_dims", [256, 128, 64]),
            dropout_rate=kwargs.get("dropout_rate", 0.3),
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
