import pandas as pd
import numpy as np
import logging
from typing import Dict
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s | AUDIT | %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class FairnessAuditor:
    """
    Профессиональный модуль аудита алгоритмической справедливости.
    Рассчитывает метрики согласно Главе 5 диссертационного исследования.
    """
    def __init__(self, dataframe: pd.DataFrame, target_col: str, protected_col: str):
        """
        :param dataframe: Данные с предсказаниями и реальными метками.
        :param target_col: Имя колонки с предсказаниями (или реальными метками).
        :param protected_col: Имя колонки защищаемого признака (1 - privileged, 0 - unprivileged).
        """
        self.df = dataframe.copy()
        self.y = target_col
        self.z = protected_col
        
        # Проверка наличия колонок
        if self.y not in self.df.columns or self.z not in self.df.columns:
            raise ValueError(f"Колонки {self.y} или {self.z} не найдены в данных.")

    def get_full_report(self, y_true_col: str = None) -> Dict[str, float]:
        """
        Генерирует полный отчет по метрикам справедливости и точности.
        """
        # Разделение на группы
        group_p = self.df[self.df[self.z] == 1] # Привилегированная (напр. мужчины)
        group_u = self.df[self.df[self.z] == 0] # Непривилегированная (напр. женщины)

        if len(group_p) == 0 or len(group_u) == 0:
            logger.warning("Одна из групп пуста. Расчет метрик невозможен.")
            return {}

        # 1. Selection Rates (Частота выбора положительного класса)
        sr_p = group_p[self.y].mean()
        sr_u = group_u[self.y].mean()

        # 2. Statistical Parity Difference (SPD)
        # Идеальное значение: 0
        spd = sr_u - sr_p

        # 3. Disparate Impact (DI)
        # Идеальное значение: 1.0 (Правило 4/5 гласит, что должно быть > 0.8)
        di = sr_u / sr_p if sr_p > 0 else 0.0

        metrics = {
            "accuracy": 0.0,
            "f1_score": 0.0,
            "statistical_parity_difference": round(float(spd), 4),
            "disparate_impact": round(float(di), 4),
            "selection_rate_privileged": round(float(sr_p), 4),
            "selection_rate_unprivileged": round(float(sr_u), 4),
            "equal_opportunity_difference": 0.0
        }

        # Если передана колонка с истинными значениями, считаем метрики качества
        if y_true_col and y_true_col in self.df.columns:
            y_true = self.df[y_true_col]
            y_pred = self.df[self.y]
            
            metrics["accuracy"] = round(accuracy_score(y_true, y_pred), 4)
            metrics["f1_score"] = round(f1_score(y_true, y_pred), 4)
            
            # Equal Opportunity Difference (Разница в True Positive Rate)
            # Измеряет, насколько одинаково модель находит "хороших" кандидатов в обеих группах
            tpr_p = self._get_tpr(group_p, y_true_col)
            tpr_u = self._get_tpr(group_u, y_true_col)
            metrics["equal_opportunity_difference"] = round(tpr_u - tpr_p, 4)

        return metrics

    def _get_tpr(self, group_df: pd.DataFrame, y_true_col: str) -> float:
        """Вспомогательный метод для расчета True Positive Rate."""
        y_true = group_df[y_true_col]
        y_pred = group_df[self.y]
        
        # Если истинных позитивных исходов нет, TPR не определен
        if sum(y_true) == 0:
            return 0.0
            
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        return tp / (tp + fn)

if __name__ == "__main__":
    # Тестовый пример
    test_data = pd.DataFrame({
        'pred': [1, 1, 0, 0, 1, 0, 1, 1],
        'true': [1, 0, 0, 0, 1, 1, 1, 0],
        'sex':  [1, 1, 1, 1, 0, 0, 0, 0] # 4 мужчины, 4 женщины
    })
    
    auditor = FairnessAuditor(test_data, 'pred', 'sex')
    report = auditor.get_full_report(y_true_col='true')
    
    print("\n--- ТЕСТОВЫЙ ОТЧЕТ АУДИТА ---")
    for k, v in report.items():
        print(f"{k.upper():<30}: {v}")