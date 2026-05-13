import os
import pickle
import torch
from security_manager import SecurityManager, SafeUnpickler

# ---------------------------------------------------------
# СЦЕНАРИЙ 1: Имитация атаки RCE (Pickle Bomb)
# ---------------------------------------------------------
class MaliciousPayload(object):
    """Вредоносный класс, который пытается выполнить системную команду"""
    def __reduce__(self):
        return (os.system, ('echo "HACKED! ВАША СИСТЕМА ВЗЛОМАНА!"',))

print("\n=== ТЕСТ 1: ЗАЩИТА ОТ RCE (PICKLE BOMB) ===")
malicious_file = "models/hacked_model.pth"

# 1. Создаем "зараженный" файл модели
os.makedirs("models", exist_ok=True)
with open(malicious_file, "wb") as f:
    pickle.dump(MaliciousPayload(), f)

print(f"[!] Вредоносный файл создан: {malicious_file}")
print("[*] Попытка безопасной десериализации (SafeUnpickler)...")

# 2. Пытаемся его загрузить через НАШ парсер напрямую
try:
    with open(malicious_file, 'rb') as f:
        SafeUnpickler(f).load()
except pickle.UnpicklingError as e:
    print(f"[+] АТАКА ОТБИТА! Ошибка: {e}")


# ---------------------------------------------------------
# СЦЕНАРИЙ 2: Имитация подмены весов (Model Tampering)
# ---------------------------------------------------------
print("\n=== ТЕСТ 2: КОНТРОЛЬ ЦЕЛОСТНОСТИ (SHA-256) ===")
fake_model_file = "models/fake_weights.pth"

# 1. Создаем фейковую модель
with open(fake_model_file, "wb") as f:
    f.write(b"fake_tensor_weights_123")

# 2. Злоумышленник знает, что система ждет вот этот эталонный хеш
expected_hash = "d38f34a4c6b92d3b58388babd74b4e79ee18d237765ff568aabde6439a42393c"

print(f"[*] Попытка загрузить измененный файл с эталонным хешем...")
try:
    SecurityManager.verify(fake_model_file, expected=expected_hash)
except ValueError as e:
    print(f"[+] АТАКА ОТБИТА! Ошибка: {e}")
