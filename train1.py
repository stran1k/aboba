"""
train1.py — Обучение модели GazeNet на датасете MPIIGaze.

Стратегия: Leave-One-Person-Out (LOPO) — обучение на p00-p13, тест на p14.
Функция потерь: Angular Loss (угловая ошибка в радианах).
Оптимизатор: AdamW с warmup 3 эпохи + cosine decay.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import random
import os
import matplotlib
matplotlib.use('Agg')   # без GUI — для сохранения в файл
import matplotlib.pyplot as plt

from dataset_loader import GazeDataset
from test_model import GazeNet

# ───────────────────────────────────────────────
# КОНФИГУРАЦИЯ
# ───────────────────────────────────────────────
DATA_PATH     = r'C:\Users\User\Desktop\def\ai test\1\MPIIGaze\Data\Normalized'
MODEL_SAVE    = "gaze_model_lopo.pth"   # файл для сохранения лучших весов

TRAIN_IDS     = [f'p{i:02d}' for i in range(14)]  # p00 - p13
TEST_ID       = ['p14']                            # оставляем одного для теста

BATCH_SIZE    = 256    # больше батч = стабильнее градиенты
LEARNING_RATE = 1e-3   # стартуем высоко, scheduler опустит до 1e-6
EPOCHS        = 50
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Воспроизводимость результатов
torch.backends.cudnn.benchmark = True
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

print(f"Устройство: {DEVICE}", flush=True)
print(f"Обучаем на: {len(TRAIN_IDS)} чел., тестируем на: {TEST_ID}", flush=True)


# ───────────────────────────────────────────────
# ANGULAR LOSS
# ───────────────────────────────────────────────
def pitchyaw_to_vector(pitchyaw):
    """
    Конвертирует углы (pitch, yaw) в 3D единичный вектор направления взгляда.

    Args:
        pitchyaw (torch.Tensor): Тензор (B, 2) — [pitch, yaw] в радианах.

    Returns:
        torch.Tensor: Тензор (B, 3) — единичные векторы [x, y, z].
    """
    pitch = pitchyaw[:, 0]
    yaw   = pitchyaw[:, 1]
    x = -torch.cos(pitch) * torch.sin(yaw)
    y = -torch.sin(pitch)
    z = -torch.cos(pitch) * torch.cos(yaw)
    return torch.stack((x, y, z), dim=1)


def angular_loss(pred, label):
    """
    Вычисляет угловую ошибку между предсказанием и меткой в радианах.

    Конвертирует pitch/yaw в 3D векторы, нормализует, считает arccos dot product.
    Используется вместо MSE — корректно работает с угловыми координатами.

    Args:
        pred  (torch.Tensor): Предсказания модели (B, 2).
        label (torch.Tensor): Метки (B, 2).

    Returns:
        torch.Tensor: Скалярное значение средней угловой ошибки в радианах.
    """
    pred_vec  = torch.nn.functional.normalize(pitchyaw_to_vector(pred),  dim=1)
    label_vec = torch.nn.functional.normalize(pitchyaw_to_vector(label), dim=1)
    # Clamp предотвращает NaN в arccos при значениях ровно ±1
    dot = torch.sum(pred_vec * label_vec, dim=1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.mean(torch.acos(dot))


def compute_angular_error(pred, label):
    """
    Вычисляет угловую ошибку в градусах (для логирования).

    Args:
        pred  (torch.Tensor): Предсказания модели (B, 2).
        label (torch.Tensor): Метки (B, 2).

    Returns:
        float: Средняя угловая ошибка в градусах.
    """
    pred_vec  = torch.nn.functional.normalize(pitchyaw_to_vector(pred),  dim=1)
    label_vec = torch.nn.functional.normalize(pitchyaw_to_vector(label), dim=1)
    dot = torch.sum(pred_vec * label_vec, dim=1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.mean(torch.acos(dot) * (180 / np.pi)).item()


# ───────────────────────────────────────────────
# АУГМЕНТАЦИИ
# ───────────────────────────────────────────────
def augment_batch(images):
    """
    Применяет случайные аугментации к батчу изображений (на GPU).

    Аугментации имитируют реальные условия: разное освещение, камеры,
    расфокус. Все операции выполняются на тензорах без копирования на CPU.

    Args:
        images (torch.Tensor): Батч изображений (B, 1, 36, 60), [0, 1].

    Returns:
        torch.Tensor: Аугментированный батч (B, 1, 36, 60), [0, 1].
    """
    B = images.size(0)

    # Яркость ±40% — имитирует разное освещение
    brightness = torch.empty(B, 1, 1, 1, device=images.device).uniform_(0.6, 1.4)
    images = images * brightness

    # Сдвиг контраста — имитирует разные настройки камеры
    contrast = torch.empty(B, 1, 1, 1, device=images.device).uniform_(-0.15, 0.15)
    images = images + contrast

    # Гауссовый шум — имитирует шум сенсора камеры
    noise = torch.randn_like(images) * 0.03
    images = images + noise

    # Горизонтальное размытие (30% шанс) — имитирует расфокус
    if random.random() < 0.3:
        kernel = torch.ones(1, 1, 1, 3, device=images.device) / 3
        images = torch.nn.functional.conv2d(images, kernel, padding=(0, 1), groups=1)

    return images.clamp(0, 1)  # обрезаем выход за пределы [0, 1]


def augment_with_flip(images, labels):
    """
    Случайный горизонтальный флип с корректировкой метки yaw.

    При зеркальном отражении глаза горизонтальный угол взгляда
    меняет знак: yaw -> -yaw. Без этой корректировки аугментация
    добавляла бы неверные обучающие примеры.

    Args:
        images (torch.Tensor): Батч изображений (B, 1, 36, 60).
        labels (torch.Tensor): Батч меток (B, 2) — [pitch, yaw].

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Аугментированные images и labels.
    """
    B = images.size(0)
    flip_mask = torch.rand(B) < 0.3  # 30% изображений переворачиваем

    images[flip_mask] = torch.flip(images[flip_mask], dims=[3])  # flip по ширине
    labels[flip_mask, 1] = -labels[flip_mask, 1]  # yaw меняет знак

    return images, labels


# ───────────────────────────────────────────────
# ОБУЧЕНИЕ
# ───────────────────────────────────────────────
def train_cross_subject():
    """
    Основная функция обучения модели GazeNet.

    Стратегия LOPO: p00-p13 для обучения, p14 для теста.
    Сохраняет лучшие веса и строит график обучения после завершения.
    """
    print("\n=== ЗАГРУЗКА ДАННЫХ ===", flush=True)
    train_dataset = GazeDataset(DATA_PATH, TRAIN_IDS)
    test_dataset  = GazeDataset(DATA_PATH, TEST_ID)

    # shuffle=True для обучения — перемешиваем каждую эпоху
    # pin_memory=True ускоряет передачу данных на GPU
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=True)

    print(f"Train: {len(train_dataset)}  |  Test: {len(test_dataset)}", flush=True)

    if len(train_dataset) == 0:
        print("❌ Train датасет пуст!", flush=True)
        return

    # Инициализация модели
    model = GazeNet().to(DEVICE)
    print("Обучаем с нуля", flush=True)

    # AdamW — Adam с правильным weight decay (не добавляется к градиенту)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)

    def lr_lambda(epoch):
        """Warmup 3 эпохи линейно, затем cosine decay до eta_min."""
        warmup = 3
        if epoch < warmup:
            return (epoch + 1) / warmup  # линейный рост 0 -> 1
        progress = (epoch - warmup) / (EPOCHS - warmup)
        return 0.5 * (1 + np.cos(np.pi * progress))  # cosine decay 1 -> 0

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Санитарная проверка модели перед обучением
    dummy = torch.randn(2, 1, 36, 60).to(DEVICE)
    with torch.no_grad():
        test_out = model(dummy)
    print(f"Тест модели: shape={test_out.shape}, NaN={torch.isnan(test_out).any()}", flush=True)
    print("======================\n", flush=True)

    best_error   = float('inf')  # лучшая угловая ошибка на тесте
    no_improve   = 0             # счётчик эпох без улучшения
    patience     = 10            # early stopping через 10 эпох без улучшений
    train_losses = []            # история train loss для графика
    test_errors  = []            # история test error для графика

    for epoch in range(EPOCHS):
        start = time.time()
        model.train()  # включаем dropout и batchnorm в режим обучения
        train_loss    = 0.0
        valid_batches = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Пропускаем батчи с NaN (могут возникнуть из-за грязных данных)
            if torch.isnan(images).any() or torch.isnan(labels).any():
                continue

            # Применяем аугментации
            images = augment_batch(images)
            images, labels = augment_with_flip(images, labels)

            optimizer.zero_grad()
            outputs = model(images)

            # Пропускаем батч если модель выдала NaN
            if torch.isnan(outputs).any():
                continue

            loss = angular_loss(outputs, labels)

            # Пропускаем если loss нестабилен
            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            # Gradient clipping — предотвращает взрывной рост градиентов
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss    += loss.item()
            valid_batches += 1

        if valid_batches == 0:
            continue

        avg_train_loss = train_loss / valid_batches
        scheduler.step()  # обновляем learning rate

        # ── ОЦЕНКА НА ТЕСТОВОМ УЧАСТНИКЕ ──
        model.eval()  # отключаем dropout для инференса
        test_error = 0.0
        batches    = 0

        with torch.no_grad():  # не считаем градиенты при тесте
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                err = compute_angular_error(outputs, labels)
                if not np.isnan(err):
                    test_error += err
                    batches    += 1

        avg_test_error = test_error / batches if batches > 0 else float('inf')
        lr_now = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
              f"Loss: {avg_train_loss:.4f} rad | "
              f"Test: {avg_test_error:.2f}° | "
              f"LR: {lr_now:.2e} | "
              f"Time: {time.time()-start:.1f}s", flush=True)

        # Сохраняем историю для графика
        train_losses.append(avg_train_loss)
        test_errors.append(avg_test_error)

        # Сохраняем веса если улучшились
        if avg_test_error < best_error:
            best_error = avg_test_error
            no_improve = 0
            torch.save(model.state_dict(), MODEL_SAVE)
            print(f"  ✓ Лучшая модель сохранена ({best_error:.2f}°)", flush=True)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"⏹ Early stopping на эпохе {epoch+1}", flush=True)
                break

    print(f"\n🎉 Готово. Лучшая угловая ошибка: {best_error:.2f}°", flush=True)

    # ── ГРАФИК ОБУЧЕНИЯ ──
    epochs_range = list(range(1, len(train_losses) + 1))
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Train loss — левая ось (синяя)
    color_train = '#2196F3'
    ax1.set_xlabel('Эпоха', fontsize=13)
    ax1.set_ylabel('Train Loss (рад)', color=color_train, fontsize=13)
    ax1.plot(epochs_range, train_losses, color=color_train, linewidth=2.5,
             marker='o', markersize=5, label='Train Loss')
    ax1.tick_params(axis='y', labelcolor=color_train)

    # Test error — правая ось (красная)
    ax2 = ax1.twinx()
    color_test = '#F44336'
    ax2.set_ylabel('Test Angular Error (°)', color=color_test, fontsize=13)
    ax2.plot(epochs_range, test_errors, color=color_test, linewidth=2.5,
             marker='s', markersize=5, linestyle='--', label='Test Error')
    ax2.tick_params(axis='y', labelcolor=color_test)

    # Вертикальная линия на лучшей эпохе
    best_ep  = test_errors.index(min(test_errors)) + 1
    best_val = min(test_errors)
    ax2.axvline(x=best_ep, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
    ax2.annotate(f'Лучшая эпоха {best_ep}\n{best_val:.2f}°',
                 xy=(best_ep, best_val), xytext=(best_ep + 0.4, best_val + 0.1),
                 fontsize=10, color='green',
                 arrowprops=dict(arrowstyle='->', color='green'))

    # Объединяем легенды двух осей
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=11)

    plt.title('GazeNet: Train Loss и Test Angular Error по эпохам', fontsize=14)
    plt.xticks(epochs_range)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("📊 График сохранён: training_curves.png", flush=True)


if __name__ == "__main__":
    train_cross_subject()