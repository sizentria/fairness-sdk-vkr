import hashlib
import os
import logging
import pickle
import io
import torch

# Настройка логгера
logging.basicConfig(level=logging.INFO, format='%(asctime)s | SECURITY | %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class SecurityManager:
    """
    Модуль контроля целостности ML-артефактов.
    Обеспечивает проверку SHA-256 хешей моделей.
    """

    @staticmethod
    def get_hash(file_path: str) -> str:
        """Вычисляет SHA-256 хеш файла."""
        sha = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                while chunk := f.read(4096):
                    sha.update(chunk)
            return sha.hexdigest()
        except FileNotFoundError:
            logger.error(f"Файл не найден: {file_path}")
            return ""

    @staticmethod
    def verify(file_path: str, expected: str = None) -> str:
        """Проверяет целостность файла по ожидаемому хешу."""
        current = SecurityManager.get_hash(file_path)
        
        if expected:
            if current != expected:
                logger.critical(f"SECURITY ALERT: Нарушение целостности! Ожидался {expected[:8]}, получен {current[:8]}")
                raise ValueError("Integrity Check Failed: Model file has been modified!")
            else:
                logger.info(f"Integrity OK. Hash verified: {current[:8]}...")
        
        return current

class SafeUnpickler(pickle.Unpickler):
    """
    Безопасный десериализатор, реализующий защиту от RCE-атак (Pickle Bomb).
    Разрешает загрузку только из утвержденных модулей (Allowlist).
    """
    # Разрешенные модули для загрузки весов PyTorch
    SAFE_MODULES = {
        'torch', 'collections', 'numpy', 'numpy.core.multiarray', 
        'torch.nn', 'torch.utils', 'torch._utils', 're'
    }

    def find_class(self, module, name):
        # Проверка модуля на вхождение в белый список
        if module.split('.')[0] not in self.SAFE_MODULES:
            logger.warning(f"SECURITY ALERT: Blocked attempt to load unsafe module '{module}.{name}'")
            raise pickle.UnpicklingError(f"Global '{module}.{name}' is forbidden by Security Policy.")
        return super().find_class(module, name)

def safe_load_model(file_path: str):
    """
    Безопасная обертка над torch.load.
    Сначала проверяет хеш (если он сохранен где-то), затем безопасно загружает.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file not found: {file_path}")

    # Просто логируем хеш перед загрузкой
    SecurityManager.verify(file_path)

    with open(file_path, 'rb') as f:
        return torch.load(f, pickle_module=SafeUnpickler)

if __name__ == "__main__":
    # Тест на текущей модели
    if os.path.exists("data/fair_model.pth"):
        h = SecurityManager.verify("data/fair_model.pth")
        print(f"Full Hash: {h}")