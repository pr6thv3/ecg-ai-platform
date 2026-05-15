from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    """Compact 1D CNN baseline for fixed-window ECG beat classification."""

    def __init__(self, num_classes: int = 5, input_size: int = 360, dropout: float = 0.25):
        super().__init__()
        _validate_model_args(num_classes, input_size)
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _ensure_channels(x)
        x = self.features(x).reshape(x.size(0), -1)
        return self.classifier(x)


ECGNet = BaselineCNN


class ResidualBlock1D(nn.Module):
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        padding = dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class ResNet1D(nn.Module):
    def __init__(self, num_classes: int = 5, input_size: int = 360, dropout: float = 0.25):
        super().__init__()
        _validate_model_args(num_classes, input_size)
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            ResidualBlock1D(64, dilation=1),
            nn.MaxPool1d(2),
            ResidualBlock1D(64, dilation=2),
            nn.MaxPool1d(2),
            ResidualBlock1D(64, dilation=4),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            ResidualBlock1D(128, dilation=2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(float(dropout)), nn.Linear(128, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _ensure_channels(x)
        return self.classifier(self.blocks(self.stem(x)))


class InceptionBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        branch = out_channels // 4
        self.bottleneck = nn.Conv1d(in_channels, branch, kernel_size=1)
        self.conv3 = nn.Conv1d(branch, branch, kernel_size=3, padding=1)
        self.conv9 = nn.Conv1d(branch, branch, kernel_size=9, padding=4)
        self.conv19 = nn.Conv1d(branch, branch, kernel_size=19, padding=9)
        self.pool = nn.Sequential(nn.MaxPool1d(3, stride=1, padding=1), nn.Conv1d(in_channels, branch, kernel_size=1))
        self.bn = nn.BatchNorm1d(branch * 4)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.bottleneck(x)
        y = torch.cat([self.conv3(z), self.conv9(z), self.conv19(z), self.pool(x)], dim=1)
        return self.activation(self.bn(y))


class InceptionTime1D(nn.Module):
    def __init__(self, num_classes: int = 5, input_size: int = 360, dropout: float = 0.25):
        super().__init__()
        _validate_model_args(num_classes, input_size)
        self.net = nn.Sequential(
            InceptionBlock1D(1, 64),
            InceptionBlock1D(64, 64),
            InceptionBlock1D(64, 128),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(float(dropout)),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(_ensure_channels(x))


class CNNLSTM(nn.Module):
    def __init__(self, num_classes: int = 5, input_size: int = 360, dropout: float = 0.25):
        super().__init__()
        _validate_model_args(num_classes, input_size)
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.rnn = nn.LSTM(input_size=64, hidden_size=64, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(nn.Dropout(float(dropout)), nn.Linear(128, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(_ensure_channels(x)).transpose(1, 2)
        _, (hidden, _) = self.rnn(x)
        features = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.classifier(features)


MODEL_REGISTRY: dict[str, Callable[[int, int, float], nn.Module]] = {
    "baseline_cnn": lambda num_classes, input_size, dropout: BaselineCNN(
        num_classes=num_classes, input_size=input_size, dropout=dropout
    ),
    "resnet1d": lambda num_classes, input_size, dropout: ResNet1D(
        num_classes=num_classes, input_size=input_size, dropout=dropout
    ),
    "inceptiontime": lambda num_classes, input_size, dropout: InceptionTime1D(
        num_classes=num_classes, input_size=input_size, dropout=dropout
    ),
    "cnn_lstm": lambda num_classes, input_size, dropout: CNNLSTM(
        num_classes=num_classes, input_size=input_size, dropout=dropout
    ),
}


def build_model(model_type: str, num_classes: int, input_size: int, dropout: float = 0.25) -> nn.Module:
    normalized = model_type.lower()
    if normalized not in MODEL_REGISTRY:
        supported = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unsupported model.type '{model_type}'. Supported values: {supported}.")
    return MODEL_REGISTRY[normalized](num_classes, input_size, float(dropout))


def _ensure_channels(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 2:
        x = x.unsqueeze(1)
    if x.ndim != 3:
        raise ValueError(f"Expected ECG tensor with 2 or 3 dimensions, got shape {tuple(x.shape)}.")
    return x


def _validate_model_args(num_classes: int, input_size: int) -> None:
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2.")
    if input_size < 32:
        raise ValueError("input_size must be at least 32 samples.")
