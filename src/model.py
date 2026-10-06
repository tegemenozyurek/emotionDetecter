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
