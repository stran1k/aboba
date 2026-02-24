import os
import scipy.io
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class GazeDataset(Dataset):
    def __init__(self, data_dir, participant_ids, transform=None):
        """
        Args:
            data_dir (str): Путь к папке Data/Normalized
            participant_ids (list): Список ID участников, например ['p00', 'p01']
            transform (callable, optional): Аугментации
        """
        self.transform = transform
        self.images = []
        self.labels = []  # [pitch, yaw]
        
        print(f"--- Инициализация датасета для: {participant_ids} ---")
        
        for pid in participant_ids:
            # Путь к папке участника: Data/Normalized/p00
            pid_dir = os.path.join(data_dir, pid)
            
            if not os.path.isdir(pid_dir):
                print(f"[Warning] Папка участника не найдена: {pid_dir}")
                continue
            
            # Перебираем все файлы .mat внутри папки (day01.mat, day02.mat ...)
            mat_files = [f for f in os.listdir(pid_dir) if f.endswith('.mat')]
            
            if not mat_files:
                print(f"[Warning] В папке {pid} нет .mat файлов.")
                continue
                
            print(f"Загрузка участника {pid}: найдено {len(mat_files)} файлов.")
            
            for mat_file in mat_files:
                full_path = os.path.join(pid_dir, mat_file)
                self._load_mat_file(full_path)
            
        # Конвертация в numpy
        if len(self.images) > 0:
            self.images = np.array(self.images, dtype=np.float32)
            self.labels = np.array(self.labels, dtype=np.float32)
            
            # Добавляем канал: (N, 36, 60) -> (N, 1, 36, 60)
            self.images = np.expand_dims(self.images, axis=1)
            print(f"Успешно загружено всего: {len(self.images)} сэмплов.")
        else:
            print("ОШИБКА: Данные не были загружены. Проверьте пути.")

    def _load_mat_file(self, path):
        try:
            # squeeze_me=True критичен для удаления лишних размерностей MATLAB
            mat = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
            
            # Структура MPIIGaze normalized: data -> right/left -> image/gaze
            if 'data' not in mat:
                return

            data = mat['data']
            
            # Проверяем наличие полей right/left
            # В зависимости от версии scipy.io и формата mat, доступ может отличаться
            eyes_list = []
            
            if hasattr(data, 'right'): eyes_list.append(data.right)
            if hasattr(data, 'left'): eyes_list.append(data.left)
            
            # Если атрибутов нет, возможно data - это словарь (реже)
            if not eyes_list and isinstance(data, dict):
                if 'right' in data: eyes_list.append(data['right'])
                if 'left' in data: eyes_list.append(data['left'])

            for eye in eyes_list:
                # eye.image: (N, 36, 60), eye.gaze: (N, 3)
                # Иногда бывает, что eye.image - это просто (36, 60) если N=1
                imgs = eye.image
                gaze = eye.gaze
                
                # Приводим к batch-виду, если там всего 1 кадр
                if imgs.ndim == 2:
                    imgs = imgs[np.newaxis, ...]
                    gaze = gaze[np.newaxis, ...]
                
                # Нормализация
                imgs = imgs / 255.0
                
                # Вектор -> Углы
                angles = self._vector_to_pitchyaw(gaze)
                
                # Добавляем в общий список.
                # Важно: используем extend для добавления батча, а не append
                for i in range(len(imgs)):
                    self.images.append(imgs[i])
                    self.labels.append(angles[i])
                
        except Exception as e:
            print(f"Ошибка чтения {os.path.basename(path)}: {e}")

    def _vector_to_pitchyaw(self, vectors):
        """
        Конвертирует 3D вектор (x, y, z) -> 2D (pitch, yaw)
        """
        n = vectors.shape[0]
        out = np.zeros((n, 2))
        
        # Нормализация вектора (на всякий случай)
        norm = np.linalg.norm(vectors, axis=1)
        # Избегаем деления на ноль
        norm = np.maximum(norm, 1e-6)
        vectors = vectors / norm[:, None]
        
        # Pitch = arcsin(-y)
        out[:, 0] = np.arcsin(-vectors[:, 1])
        # Yaw = arctan2(-x, -z)
        out[:, 1] = np.arctan2(-vectors[:, 0], -vectors[:, 2])
        
        return out

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if idx >= len(self.images):
             raise IndexError(f"Index {idx} out of range (len={len(self.images)})")
             
        img = torch.from_numpy(self.images[idx])
        label = torch.from_numpy(self.labels[idx])
        
        if self.transform:
            img = self.transform(img)
            
        return img, label

# --- Тест ---
if __name__ == "__main__":
    # Вставляем ваш путь (обратите внимание на r перед кавычками)
    data_path = r'C:\Users\Starb\Desktop\projects\cjmp\MPIIGaze\Data\Normalized' 
    
    # Тестируем на одном участнике
    dataset = GazeDataset(data_path, ['p00'])
    
    if len(dataset) > 0:
        loader = DataLoader(dataset, batch_size=16, shuffle=True)
        images, labels = next(iter(loader))
        print(f"\n[OK] Размер батча: {images.shape}") # Должно быть torch.Size([16, 1, 36, 60])
        print(f"[OK] Пример метки: {labels[0]}")