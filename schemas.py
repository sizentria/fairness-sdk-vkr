from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict
from datetime import datetime

# --- Схемы для Обучения ---

class TrainingRequest(BaseModel):
    """
    Запрос на запуск состязательного обучения (Глава 4).
    Содержит гиперпараметры для метода MINE.
    """
    dataset_path: str = Field(..., description="Путь к CSV файлу с данными Adult Census")
    features: List[str] = Field(
        ..., 
        min_length=1, 
        description="Список признаков (например: age, education-num, и т.д.)"
    )
    target: str = Field(..., description="Целевая переменная (например: target)")
    protected: str = Field(..., description="Защищаемый признак (например: sex_binary)")
    
    # ИСПРАВЛЕНИЕ: Расширен диапазон валидации для поддержки lambda=15.0 (как в эксперименте)
    # Default установлен в 15.0 как рекомендованное значение
    lambda_reg: float = Field(default=15.0, ge=0.0, le=50.0, description="Коэффициент MINE (λ)")
    
    epochs: int = Field(default=20, ge=1, le=1000, description="Количество эпох")

    @field_validator('dataset_path')
    @classmethod
    def validate_csv_extension(cls, v: str) -> str:
        if not v.lower().endswith('.csv'):
            raise ValueError("Файл данных должен быть в формате .csv")
        return v

# --- Схемы для Аудита ---

class FairnessMetrics(BaseModel):
    """Метрики алгоритмической справедливости (Глава 5)."""
    accuracy: float
    f1_score: float
    statistical_parity_difference: float
    disparate_impact: float
    equal_opportunity_difference: float
    selection_rate_privileged: float
    selection_rate_unprivileged: float

class AuditRequest(BaseModel):
    """Запрос на проведение полного технического аудита модели."""
    model_path: str = Field(..., description="Путь к сохраненному файлу .pth")
    dataset_path: str = Field(..., description="Путь к данным для теста")
    features: List[str]
    target: str
    protected: str
    expected_hash: Optional[str] = Field(None, description="SHA-256 хеш для проверки целостности")

# --- Схемы Ответов (API Response) ---

class TaskResponse(BaseModel):
    """Ответ API при успешной постановке задачи в очередь Celery."""
    task_id: str = Field(..., description="ID задачи для отслеживания статуса")
    status: str = Field(default="PENDING")
    created_at: datetime = Field(default_factory=datetime.now)

class AuditResult(BaseModel):
    """Итоговый отчет, который возвращается после завершения аудита."""
    status: str
    metrics: FairnessMetrics
    model_hash: str
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())

# --- Схема для Инференса (Предсказания) ---

class PredictionInput(BaseModel):
    """Входные данные для получения предсказания от живой модели."""
    model_id: str = Field(..., description="Имя файла модели в директории data/")
    features: Dict[str, float] = Field(..., description="Словарь признаков: {'age': 25, ...}")
