import os
import time
import logging
import torch
import pandas as pd
from typing import Dict, Any
from celery import Celery
import json

# Импорт локальных модулей (без префиксов app.core)
from fair_train_mine import MINEAdversarialTrainer, Predictor
from auditor import FairnessAuditor
from security_manager import SecurityManager

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s | WORKER | %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Инициализация Celery
# Broker и Backend указывают на локальный Redis
celery_app = Celery(
    "tasks",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0"
)

# Конфигурация специально для Windows и безопасности
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    worker_pool='solo',  # ВАЖНО: Решает проблему зависания на Windows
    task_track_started=True
)

@celery_app.task(bind=True, name="tasks.train_fair_model")
def train_fair_model_task(self, data_path: str, params: Dict[str, Any]):
    """
    Задача: Обучение модели с MINE-регуляризацией.
    Реализует принцип Accountability: модель подписывается хешем сразу после создания.
    """
    try:
        logger.info(f"Начало задачи обучения. Данные: {data_path}")
        self.update_state(state='PROGRESS', meta={'status': 'Загрузка и препроцессинг данных...'})
        
        # 1. Загрузка данных
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Файл {data_path} не найден.")
        
        df = pd.read_csv(data_path)
        
        # Извлечение параметров
        features = params['features']
        target = params['target']
        protected = params['protected']
        
        X = df[features].values
        y = df[target].values
        z = df[protected].values
        
        # 2. Инициализация тренера (Глава 4: Состязательное обучение)
        logger.info("Инициализация MINE Trainer...")
        trainer = MINEAdversarialTrainer(
            input_dim=len(features),
            z_dim=1,
            lambda_reg=params.get("lambda_reg", 0.5)
        )

        self.update_state(state='PROGRESS', meta={'status': 'Обучение нейросети (MINE)...'})
        
        # 3. Запуск цикла обучения
        history = trainer.fit(X, y, z, epochs=params.get("epochs", 20)) 

        # НОВОЕ: Сохранение истории обучения для графиков (Блок Б)
        os.makedirs("reports", exist_ok=True)
        history_path = os.path.join("reports", f"history_{int(time.time())}.json")
        with open(history_path, "w") as f:
            json.dump(history, f)
        logger.info(f"История обучения сохранена в {history_path}")

        # 4. Сохранение и Цифровая подпись (Integrity Control)
        os.makedirs("models", exist_ok=True)
        model_name = f"fair_model_{int(time.time())}.pth"
        model_path = os.path.join("models", model_name)
        
        trainer.save_model(model_path)
        
        # Генерируем хеш для гарантии неизменности артефакта
        model_hash = SecurityManager.get_hash(model_path)
        logger.info(f"Модель сохранена и подписана. Hash: {model_hash}")

        return {
            "status": "COMPLETED",
            "model_path": model_path,
            "model_hash": model_hash,
            "final_loss": history["task_loss"][-1] if history["task_loss"] else 0
        }

    except Exception as exc:
        logger.error(f"Критическая ошибка обучения: {exc}")
        # Пробрасываем исключение, чтобы Celery пометил задачу как FAILURE
        raise exc

@celery_app.task(name="tasks.run_audit")
def run_audit_task(model_path: str, dataset_path: str, config: Dict[str, Any]):
    """
    Задача: Аудит справедливости.
    Реализует принцип Resilience: безопасная загрузка и проверка целостности.
    """
    try:
        logger.info(f"Начат аудит модели: {model_path}")

        # 1. ПРОВЕРКА ЦЕЛОСТНОСТИ (Security Layer)
        expected_hash = config.get("expected_hash")
        if expected_hash:
            logger.info("Проверка цифровой подписи модели...")
            SecurityManager.verify(model_path, expected_hash)
        
        # 2. Безопасная загрузка весов
        # weights_only=True защищает от исполнения произвольного кода (pickle exploit)
        state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
        
        input_dim = len(config['features'])
        model = Predictor(input_dim)
        model.load_state_dict(state_dict)
        model.eval()

        # 3. Подготовка данных для аудита
        df = pd.read_csv(dataset_path)
        X_tensor = torch.FloatTensor(df[config['features']].values)
        
        with torch.no_grad():
            # Получаем бинарные предсказания
            preds = (torch.sigmoid(model(X_tensor)) > 0.5).int().numpy().flatten()

        # 4. Расчет метрик (Глава 5: Метрология справедливости)
        audit_df = pd.DataFrame({
            'prediction': preds,
            config['protected']: df[config['protected']].values
        })
        
        # Если есть истинные значения, добавляем их для расчета точности
        y_true_col = None
        if config['target'] in df.columns:
            audit_df['ground_truth'] = df[config['target']].values
            y_true_col = 'ground_truth'

        auditor = FairnessAuditor(audit_df, 'prediction', config['protected'])
        metrics = auditor.get_full_report(y_true_col=y_true_col)

        return {
            "status": "AUDIT_COMPLETED",
            "metrics": metrics,
            "verified": True if expected_hash else False
        }

    except Exception as exc:
        logger.error(f"Ошибка аудита: {exc}")
        return {"status": "FAILED", "error": str(exc)}