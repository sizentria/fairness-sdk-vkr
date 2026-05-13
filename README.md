# Система обеспечения справедливости и контроля целостности ИИ-моделей

## 🛠 Технологический стек
- **Core**: PyTorch (MINE-adversarial training)
- **Backend**: FastAPI
- **Task Queue**: Celery + Redis
- **Security**: SHA-256 Integrity Verification

## 📑 Научная новизна
Реализация состязательного метода минимизации взаимной информации (MINE) в распределенной среде с автоматизированным аудитом этических метрик (SPD, DI).

## 🚀 Запуск системы
1. `pip install -r requirements.txt`
2. Запуск брокера: `redis-server`
3. Запуск воркера: `celery -A tasks worker --pool=solo --loglevel=info`
4. Запуск API: `python main.py`