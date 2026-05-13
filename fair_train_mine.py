import os
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score

# Настройка логгера
logging.basicConfig(level=logging.INFO, format='%(asctime)s | MINE-CORE | %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def seed_everything(seed=42):
    """
    Обеспечение детерминизма эксперимента.
    Фиксируем RNG для воспроизводимости весов и батчинга.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Random Seed зафиксирован: {seed}")

class Predictor(nn.Module):
    """
    Основная нейросеть-классификатор (Classifier).
    
    ENGINEERING CHANGE (Глава 4.2): 
    Удален слой nn.Sigmoid(). Модель возвращает 'сырые' логиты (logits).
    Это необходимо для использования BCEWithLogitsLoss, который обеспечивает
    большую численную стабильность градиентов (LogSumExp trick).
    """
    def __init__(self, input_dim: int):
        super(Predictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2), # Увеличили Dropout для борьбы с переобучением
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

class MultiMINE(nn.Module):
    """
    Критик (Estimator) взаимной информации.
    Оценивает зависимость I(Y_pred; Z).
    """
    def __init__(self, z_dim: int):
        super(MultiMINE, self).__init__()
        # Вход: вероятность предсказания (1) + защищенный признак (z_dim)
        self.net = nn.Sequential(
            nn.Linear(1 + z_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, y_pred, z):
        return self.net(torch.cat([y_pred, z], dim=1))

class MINEAdversarialTrainer:
    """
    Оркестратор состязательного обучения (Adversarial Training Orchestrator).
    Реализует минимаксную игру: min_theta max_omega (L_task + lambda * I_MINE).
    """
    def __init__(self, input_dim: int, z_dim: int = 1, lambda_reg: float = 15.0, lr: float = 1e-3, seed: int = 42):
        seed_everything(seed)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lambda_reg = lambda_reg
        
        # Инициализация сетей
        self.predictor = Predictor(input_dim).to(self.device)
        self.critic = MultiMINE(z_dim).to(self.device)
        
        # Оптимизаторы
        self.opt_p = optim.Adam(self.predictor.parameters(), lr=lr)
        self.opt_c = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Scheduler: уменьшает LR, если Loss перестал падать
        # FIX: Убран аргумент verbose=True для совместимости с новыми версиями PyTorch
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.opt_p, mode='min', factor=0.5, patience=5
        )
        
        logger.info(f"Trainer инициализирован. Device: {self.device}. Lambda: {lambda_reg}")

    def _mine_loss(self, y_pred_prob, z):
        """
        Расчет нижней границы Донскера-Варадхана (DV-bound).
        y_pred_prob: вероятности (после sigmoid), а не логиты!
        """
        # Joint distribution (совместное)
        t_joint = self.critic(y_pred_prob, z)
        
        # Marginal distribution (маргинальное) - перемешиваем Z внутри батча
        # Перестановка индексов выполняется на GPU
        idx = torch.randperm(z.size(0), device=self.device)
        z_shuffled = z[idx]
        t_marginal = self.critic(y_pred_prob, z_shuffled)
        
        # DV-bound: E[T(x,y)] - log(E[exp(T(x,y_shuffled))])
        # Добавлено 1e-6 для численной стабильности логарифма
        mi = torch.mean(t_joint) - torch.log(torch.mean(torch.exp(t_marginal)) + 1e-6)
        return mi

    def fit(self, X: np.ndarray, y: np.ndarray, z: np.ndarray, epochs: int = 50, batch_size: int = 1024):
        """
        Основной цикл обучения.
        """
        # 1. Подготовка данных
        X_t = torch.FloatTensor(X)
        y_t = torch.FloatTensor(y).reshape(-1, 1) # Целевые метки 0/1
        z_t = torch.FloatTensor(z).reshape(-1, 1) # Защищенный признак
        
        # Расчет веса положительного класса для балансировки Loss
        num_pos = np.sum(y == 1)
        num_neg = np.sum(y == 0)
        # Защита от деления на ноль
        pos_weight_val = num_neg / max(num_pos, 1)
        
        pos_weight_tensor = torch.FloatTensor([pos_weight_val]).to(self.device)
        logger.info(f"Class Imbalance Correction: pos_weight calculated as {pos_weight_val:.4f}")

        # Критерий потерь с учетом весов
        criterion_task = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

        # Создание загрузчика данных
        dataset = TensorDataset(X_t, y_t, z_t)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        history = {"task_loss": [], "mi_loss": [], "f1_score": []}
        
        logger.info(f"Запуск обучения: {epochs} эпох, Batch Size: {batch_size}")
        self.predictor.train()
        self.critic.train()

        for epoch in range(epochs):
            epoch_task_loss = 0.0
            epoch_mi = 0.0
            all_preds = []
            all_targets = []

            for batch_idx, (bx, by, bz) in enumerate(dataloader):
                bx, by, bz = bx.to(self.device), by.to(self.device), bz.to(self.device)
                
                # --- Шаг 1: Обучение Критика (Максимизация MI) ---
                for _ in range(5):
                    self.opt_c.zero_grad()
                    with torch.no_grad():
                        logits = self.predictor(bx)
                        probs = torch.sigmoid(logits)
                    
                    mi = self._mine_loss(probs, bz)
                    loss_c = -mi 
                    loss_c.backward()
                    self.opt_c.step()

                # --- Шаг 2: Обучение Предиктора ---
                self.opt_p.zero_grad()
                logits = self.predictor(bx) 
                probs = torch.sigmoid(logits) 
                
                task_loss = criterion_task(logits, by)
                mi_val = self._mine_loss(probs, bz)
                
                total_loss = task_loss + (self.lambda_reg * mi_val)
                total_loss.backward()
                self.opt_p.step()

                # Сбор метрик
                epoch_task_loss += task_loss.item()
                epoch_mi += mi_val.item()
                
                preds_bin = (probs > 0.5).long()
                all_preds.append(preds_bin.cpu().numpy())
                all_targets.append(by.cpu().numpy())

            # --- Конец эпохи ---
            avg_task_loss = epoch_task_loss / len(dataloader)
            avg_mi = epoch_mi / len(dataloader)
            
            all_preds = np.concatenate(all_preds)
            all_targets = np.concatenate(all_targets)
            
            epoch_acc = accuracy_score(all_targets, all_preds)
            epoch_f1 = f1_score(all_targets, all_preds)
            
            history["task_loss"].append(avg_task_loss)
            history["mi_loss"].append(avg_mi)
            history["f1_score"].append(epoch_f1)

            # Шаг планировщика
            self.scheduler.step(avg_task_loss)

            if (epoch + 1) % 1 == 0:
                logger.info(
                    f"Epoch {epoch+1:02d} | "
                    f"Loss: {avg_task_loss:.4f} | "
                    f"MI: {avg_mi:.4f} | "
                    f"F1: {epoch_f1:.4f}"
                )

        return history

    def save_model(self, path: str):
        torch.save(self.predictor.state_dict(), path)
        logger.info(f"Модель сохранена: {path}")

if __name__ == "__main__":
    # Тест
    X_dummy = np.random.randn(100, 10).astype(np.float32)
    y_dummy = np.random.randint(0, 2, size=(100,)).astype(np.float32)
    z_dummy = np.random.randint(0, 2, size=(100,)).astype(np.float32)
    trainer = MINEAdversarialTrainer(input_dim=10)
    trainer.fit(X_dummy, y_dummy, z_dummy, epochs=2, batch_size=10)
