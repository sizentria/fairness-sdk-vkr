import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

# Импорт локального модуля
try:
    from fair_train_mine import MINEAdversarialTrainer
except ImportError:
    sys.exit("Ошибка: Файл fair_train_mine.py не найден.")

# --- НАСТРОЙКА РУССКОГО ЯЗЫКА ДЛЯ ГРАФИКОВ (ГОСТ) ---
sns.set_theme(style="whitegrid")
# DejaVu Sans - шрифт, который есть везде и поддерживает кириллицу
plt.rcParams.update({'font.size': 12, 'figure.dpi': 150, 'font.family': 'DejaVu Sans'})

DATA_PATH = "data/adult_processed.csv"

def get_data():
    """Загрузка и подготовка данных"""
    path = DATA_PATH
    if not os.path.exists(path):
        if os.path.exists("adult_processed.csv"):
            path = "adult_processed.csv"
        else:
            raise FileNotFoundError(f"Файл данных не найден: {DATA_PATH}")
            
    df = pd.read_csv(path).dropna()
    
    # Определение колонок
    target_col = next((c for c in ['income', 'target'] if c in df.columns), 'income')
    if df[target_col].dtype == object:
        df[target_col] = df[target_col].apply(lambda x: 1 if '>50K' in str(x) else 0)
        
    sex_col = next((c for c in ['sex', 'gender', 'sex_binary'] if c in df.columns), 'sex')
    if df[sex_col].dtype == object:
        df[sex_col] = df[sex_col].apply(lambda x: 1 if 'Male' in str(x) else 0)
        
    # Признаки
    drop = [target_col, sex_col, 'fnlwgt', 'education']
    feat_cols = [c for c in df.columns if c not in drop]
    num_df = df[feat_cols].select_dtypes(include=[np.number])
    
    X = num_df.values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)
    z = df[sex_col].values.astype(np.float32)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return train_test_split(X, y, z, test_size=0.2, stratify=y, random_state=42), num_df.shape[1]

def calc_fairness(y_pred, z_test):
    idx_priv = (z_test == 1)
    idx_unpriv = (z_test == 0)
    
    prob_priv = np.mean(y_pred[idx_priv])
    prob_unpriv = np.mean(y_pred[idx_unpriv])
    
    spd = prob_unpriv - prob_priv
    di = prob_unpriv / max(prob_priv, 1e-6)
    return spd, di

def main():
    print("--- ГЕНЕРАЦИЯ ГРАФИКОВ (RU / ГОСТ) ---")
    (X_train, X_test, y_train, y_test, z_train, z_test), input_dim = get_data()
    
    # 1. BASELINE (Без защиты)
    print("1/2 Обучение Базовой модели (Baseline)...")
    base_trainer = MINEAdversarialTrainer(input_dim, lambda_reg=0.0)
    base_trainer.fit(X_train, y_train, z_train, epochs=10, batch_size=1024)
    
    base_trainer.predictor.eval()
    with torch.no_grad():
        logits = base_trainer.predictor(torch.FloatTensor(X_test).to(base_trainer.device))
        preds_base = (torch.sigmoid(logits).cpu().numpy() > 0.5).astype(int).flatten()

    # 2. FAIRNESS MODEL (Наша модель)
    print("2/2 Обучение Защищенной модели (Fairness)...")
    # Lambda = 15.0 для высокого DI
    fair_trainer = MINEAdversarialTrainer(input_dim, lambda_reg=15.0) 
    history = fair_trainer.fit(X_train, y_train, z_train, epochs=30, batch_size=1024)
    
    fair_trainer.predictor.eval()
    with torch.no_grad():
        logits = fair_trainer.predictor(torch.FloatTensor(X_test).to(fair_trainer.device))
        preds_fair = (torch.sigmoid(logits).cpu().numpy() > 0.5).astype(int).flatten()

    # --- ГРАФИК 1: Динамика (RU) ---
    print("Рисуем график динамики...")
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Эпохи обучения')
    ax1.set_ylabel('Ошибка классификации (Task Loss)', color=color)
    ax1.plot(history['task_loss'], color=color, linewidth=2, label='Task Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Метрика F1-Score', color=color)
    ax2.plot(history['f1_score'], color=color, linewidth=2, linestyle='--', label='F1-Score')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Динамика состязательного обучения (MINE)')
    fig.tight_layout()
    plt.savefig('training_dynamics_ru.png')
    print("Сохранен: training_dynamics_ru.png")

    # --- ГРАФИК 2: Сравнение (RU) ---
    print("Рисуем график сравнения...")
    metrics = {'Модель': [], 'Метрика': [], 'Значение': []}
    
    def add_metrics(name, preds):
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        spd, di = calc_fairness(preds, z_test)
        # Перевод названий метрик для графика
        metrics['Модель'].extend([name]*4)
        metrics['Метрика'].extend(['Accuracy (Точность)', 'F1-Score', 'SPD (Смещение)', 'DI (Индекс влияния)'])
        metrics['Значение'].extend([acc, f1, abs(spd), di])
        return f1, di
        
    f1_b, di_b = add_metrics('Базовая (Baseline)', preds_base)
    f1_f, di_f = add_metrics('С защитой (Fairness)', preds_fair)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=pd.DataFrame(metrics), x='Метрика', y='Значение', hue='Модель', palette=['#95a5a6', '#2ecc71'])
    
    # Линии порогов
    plt.axhline(y=0.8, color='r', linestyle='--', alpha=0.7, label='Норматив DI (0.8)')
    
    plt.title('Сравнительный анализ эффективности алгоритмов')
    plt.legend(loc='upper right')
    plt.savefig('metrics_comparison_ru.png')
    print("Сохранен: metrics_comparison_ru.png")
    
    print("\n--- ИТОГОВЫЕ ЦИФРЫ ---")
    print(f"BASELINE: F1={f1_b:.3f}, DI={di_b:.3f}")
    print(f"OURS:     F1={f1_f:.3f}, DI={di_f:.3f}")

if __name__ == "__main__":
    main()