"""The CNN that classifies a 48x48 grayscale face into 7 emotions.

Architecture (VGG-style): 4 convolutional blocks, each one
  conv3x3 -> BatchNorm -> ReLU -> conv3x3 -> BatchNorm -> ReLU -> MaxPool -> Dropout
doubles the number of channels and halves the image size:

  1x48x48 -> 64x24x24 -> 128x12x12 -> 256x6x6 -> 512x3x3

then global average pooling + a small classifier head produce 7 scores (logits).
"""
import torch
from torch import nn


def conv_block(in_channels, out_channels, dropout):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Dropout(dropout),
    )


class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1, 64, dropout=0.1),
            conv_block(64, 128, dropout=0.2),
            conv_block(128, 256, dropout=0.3),
            conv_block(256, 512, dropout=0.3),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon GPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class ResidualBlock(nn.Module):
    """Two 3x3 convolutions plus a shortcut, so the block only has to learn a correction."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                                          nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.body(x) + self.shortcut(x))


class EmotionResNet(nn.Module):
    """ResNet-18-style network for aligned 64x64 grayscale faces (eDv1.0 family).

    1x64x64 -> stem -> 64x32x32 -> 128x16x16 -> 256x8x8 -> 512x4x4 -> pool -> 7 logits
    (channel counts shown for width=64). Trained from scratch, no pretrained weights.
    """

    def __init__(self, num_classes=7, width=64, dropout=0.3):
        super().__init__()
        w = width
        self.stem = nn.Sequential(
            nn.Conv2d(1, w // 2, 3, 1, 1, bias=False), nn.BatchNorm2d(w // 2), nn.ReLU(inplace=True),
            nn.Conv2d(w // 2, w, 3, 2, 1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True),
        )
        stages, channels = [], w
        for i, out in enumerate([w, 2 * w, 4 * w, 8 * w]):
            stride = 1 if i == 0 else 2
            stages += [ResidualBlock(channels, out, stride), ResidualBlock(out, out)]
            channels = out
        self.stages = nn.Sequential(*stages)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout),
                                  nn.Linear(channels, num_classes))

    def forward(self, x):
        return self.head(self.stages(self.stem(x)))


def build_model(arch, width=64):
    if arch == "resnet":
        return EmotionResNet(width=width)
    if arch == "vgg":
        return EmotionCNN()
    raise ValueError(f"unknown architecture {arch!r}")
