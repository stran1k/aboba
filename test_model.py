"""
test_model.py — Архитектура нейросетевой модели GazeNet.

GazeNet — свёрточная нейросеть для регрессии направления взгляда.
Принимает grayscale изображение глаза (36x60), возвращает (pitch, yaw) в радианах.
"""

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation блок — механизм внимания по каналам.

    Взвешивает важность каждого канала признаков через глобальный
    average pooling и два полносвязных слоя с bottleneck.
    Улучшает точность без значительного роста числа параметров.

    Args:
        channels (int): Количество входных каналов.
        reduction (int): Коэффициент сжатия bottleneck. По умолчанию 8.
    """

    def __init__(self, channels, reduction=8):
        super().__init__()
        # Global average pooling -> FC -> ReLU -> FC -> Sigmoid
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),          # (B, C, H, W) -> (B, C, 1, 1)
            nn.Flatten(),                      # (B, C, 1, 1) -> (B, C)
            nn.Linear(channels, channels // reduction),  # сжатие
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),  # восстановление
            nn.Sigmoid()                       # веса каналов в [0, 1]
        )

    def forward(self, x):
        """
        Применяет SE-внимание к входному тензору.

        Args:
            x (torch.Tensor): Входной тензор (B, C, H, W).

        Returns:
            torch.Tensor: Тензор (B, C, H, W) с взвешенными каналами.
        """
        # Вычисляем веса каналов и масштабируем входной тензор
        scale = self.se(x).view(x.size(0), x.size(1), 1, 1)
        return x * scale


class GazeNet(nn.Module):
    """
    Свёрточная нейросеть для предсказания направления взгляда.

    Архитектура: 3 свёрточных блока с SE-вниманием + полносвязный регрессор.
    Вход: grayscale изображение глаза (1, 36, 60).
    Выход: (pitch, yaw) в радианах.

    Схема уменьшения размера после MaxPool2d(2) x3:
        36 -> 18 -> 9 -> 4
        60 -> 30 -> 15 -> 7
        Итого: 128 * 4 * 7 = 3584 признака перед FC слоями.
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # ── Блок 1: (1, 36, 60) -> (32, 18, 30) ──
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # извлечение базовых признаков
            nn.BatchNorm2d(32),                           # нормализация активаций
            nn.ReLU(),                                    # нелинейная активация
            SEBlock(32),                                  # внимание по каналам
            nn.MaxPool2d(2),                              # уменьшение пространства в 2 раза
            nn.Dropout2d(0.1),                            # регуляризация (10%)

            # ── Блок 2: (32, 18, 30) -> (64, 9, 15) ──
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # углублённые признаки
            nn.BatchNorm2d(64),
            nn.ReLU(),
            SEBlock(64),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),                          # регуляризация (15%)

            # ── Блок 3: (64, 9, 15) -> (128, 4, 7) ──
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # высокоуровневые признаки
            nn.BatchNorm2d(128),
            nn.ReLU(),
            SEBlock(128),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),                           # регуляризация (20%)
        )

        # Полносвязный регрессор: 3584 -> 512 -> 256 -> 2
        self.regressor = nn.Sequential(
            nn.Linear(128 * 4 * 7, 512),  # 3584 признака -> 512
            nn.ReLU(),
            nn.Dropout(0.3),               # сильная регуляризация на FC
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)              # выход: [pitch, yaw] в радианах
        )

    def forward(self, x):
        """
        Прямой проход модели.

        Args:
            x (torch.Tensor): Тензор изображений (B, 1, 36, 60).

        Returns:
            torch.Tensor: Предсказания (B, 2) — [pitch, yaw] в радианах.
        """
        x = self.features(x)           # свёрточные блоки
        x = x.view(x.size(0), -1)      # flatten: (B, 128, 4, 7) -> (B, 3584)
        x = self.regressor(x)          # регрессор -> (B, 2)
        return x


if __name__ == "__main__":
    model = GazeNet()
    dummy = torch.randn(2, 1, 36, 60)
    out   = model(dummy)
    print(f"Output shape: {out.shape}")       # ожидается (2, 2)
    total = sum(p.numel() for p in model.parameters())
    print(f"Параметров: {total:,}")           # ~2.1 млн