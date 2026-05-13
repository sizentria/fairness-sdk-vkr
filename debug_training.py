import os
import sys
import logging
import pandas as pd
import numpy as np
from typing import Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Импорт локального модуля обучения
try:
    from fair_train_mine import MINEAdversarialTrainer
except ImportError:
    sys.exit("CRITICAL: Module 'fair_train_mine.py' not found. Check project structure.")

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [DEBUG] - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Конфигурация путей
DATA_DIR = "data"
FILENAME = "adult_processed.csv"

def resolve_data_path() -> str:
    """Определяет абсолютный путь к файлу данных."""
    # 1. Проверка в папке data/ относительно скрипта
    path = os.path.join(os.getcwd(), DATA_DIR, FILENAME)
    if os.path.exists(path):
        return path
    
    # 2. Проверка в текущей директории (fallback)
    path = os.path.join(os.getcwd(), FILENAME)
    if os.path.exists(path):
        return path
        
    raise FileNotFoundError(f"Dataset '{FILENAME}' not found in {os.getcwd()} or ./{DATA_DIR}/")

def preprocess_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Выполняет базовую предобработку: кодирование и нормализацию.
    Returns: X, y, z, input_dim
    """
    # Определение целевых колонок
    target_col = next((c for c in ['income', 'target', 'salary'] if c in df.columns), None)
    sex_col = next((c for c in ['sex', 'gender', 'sex_binary'] if c in df.columns), None)

    if not target_col or not sex_col:
        raise ValueError("Target or Protected attribute column not found.")

    logger.info(f"Target: '{target_col}' | Protected: '{sex_col}'")

    # Удаление пропусков
    df = df.dropna()

    # Бинаризация Target
    if df[target_col].dtype == 'object':
        df[target_col] = df[target_col].apply(lambda x: 1 if '>50K' in str(x) else 0)
    
    # Бинаризация Protected Attribute
    if df[sex_col].dtype == 'object':
        df[sex_col] = df[sex_col].apply(lambda x: 1 if 'Male' in str(x) else 0)

    # Выделение признаков (исключаем служебные)
    drop_cols = [target_col, sex_col, 'fnlwgt', 'education']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    # Фильтрация только числовых колонок
    numeric_df = df[feature_cols].select_dtypes(include=[np.number])
    X_raw = numeric_df.values
    
    y = df[target_col].values.astype(np.float32)
    z = df[sex_col].values.astype(np.float32)

    # Нормализация (StandardScaler)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw).astype(np.float32)

    return X, y, z, X.shape[1]

def main():
    logger.info("Starting Debug Pipeline...")

    # 1. Загрузка
    try:
        file_path = resolve_data_path()
        logger.info(f"Loading data from: {file_path}")
        df = pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        return

    # 2. Препроцессинг
    try:
        X, y, z, input_dim = preprocess_data(df)
        logger.info(f"Data shape: {X.shape}. Features: {input_dim}")
        logger.info(f"Class Balance (Pos): {np.mean(y):.2%}")
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        return

    # 3. Сплит
    X_train, X_test, y_train, y_test, z_train, z_test = train_test_split(
        X, y, z, test_size=0.2, stratify=y, random_state=42
    )

    # 4. Инициализация и обучение
    logger.info("Initializing MINE Trainer...")
    trainer = MINEAdversarialTrainer(
        input_dim=input_dim,
        lambda_reg=1.0, 
        lr=0.001
    )

    logger.info("Starting training loop (15 epochs)...")
    try:
        history = trainer.fit(X_train, y_train, z_train, epochs=15, batch_size=1024)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return

    # 5. Результаты
    final_f1 = history['f1_score'][-1]
    final_loss = history['task_loss'][-1]

    print("-" * 40)
    logger.info(f"FINAL METRICS (Epoch 15):")
    logger.info(f"Task Loss: {final_loss:.4f}")
    logger.info(f"F1 Score:  {final_f1:.4f}")
    print("-" * 40)

    if final_f1 > 0.60:
        logger.info("SUCCESS: Model converged. F1-score is acceptable.")
    else:
        logger.warning("WARNING: F1-score is below threshold (0.60). Check data balance or hyperparameters.")

if __name__ == "__main__":
    main()
