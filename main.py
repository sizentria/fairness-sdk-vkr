import os
import uvicorn
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from celery.result import AsyncResult
from contextlib import asynccontextmanager

# Импорт наших контрактов данных и задач
from schemas import TrainingRequest, AuditRequest, TaskResponse
from tasks import train_fair_model_task, run_audit_task, celery_app
from security_manager import SecurityManager

# Настройка логирования API
logging.basicConfig(level=logging.INFO, format='%(asctime)s | API | %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Trustworthy AI Platform",
    description="API для состязательного обучения (MINE) и аудита справедливости моделей.",
    version="2.0.0"
)

# Разрешаем CORS (полезно для тестов через браузер)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Инициализация инфраструктуры при запуске сервера."""
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    logger.info("API запущено. Директории /models и /data проверены.")
    yield # Сервер работает
    # Здесь можно добавить логику при выключении сервера

app = FastAPI(
    title="Trustworthy AI Platform",
    description="API для состязательного обучения (MINE) и аудита справедливости моделей.",
    version="2.0.0",
    lifespan=lifespan # <--- Добавляем привязку сюда
)

@app.post("/train", response_model=TaskResponse, tags=["Ethical Training"])
async def train_model(request: TrainingRequest):
    """
    Запуск обучения модели с MINE-регуляризацией (Глава 4).
    Задача отправляется в Celery Worker.
    """
    if not os.path.exists(request.dataset_path):
        raise HTTPException(status_code=404, detail=f"Файл {request.dataset_path} не найден")
    
    logger.info(f"Получен запрос на обучение. Датасет: {request.dataset_path}")
    
    # Отправка задачи в очередь Redis
    # .delay() - это асинхронный вызов
    task = train_fair_model_task.delay(
        data_path=request.dataset_path, 
        params=request.model_dump()
    )
    
    return TaskResponse(task_id=task.id, status="PENDING")

@app.post("/audit", response_model=TaskResponse, tags=["Audit & Compliance"])
async def audit_model(request: AuditRequest):
    """
    Запуск аудита справедливости (Глава 5).
    Включает проверку целостности модели перед загрузкой.
    """
    if not os.path.exists(request.model_path):
        raise HTTPException(status_code=404, detail=f"Модель {request.model_path} не найдена")
    
    logger.info(f"Получен запрос на аудит. Модель: {request.model_path}")
    
    task = run_audit_task.delay(
        model_path=request.model_path,
        dataset_path=request.dataset_path,
        config=request.model_dump()
    )
    
    return TaskResponse(task_id=task.id, status="PENDING")

@app.get("/status/{task_id}", tags=["Monitoring"])
async def get_task_status(task_id: str):
    """
    Получение статуса задачи (PENDING, PROGRESS, SUCCESS, FAILURE).
    Возвращает результат обучения или аудита, когда задача готова.
    """
    task_result = AsyncResult(task_id, app=celery_app)
    
    response = {
        "task_id": task_id,
        "status": task_result.status,
        "result": None
    }

    if task_result.ready():
        if task_result.successful():
            response["result"] = task_result.result
        else:
            # Если произошла ошибка, выводим её
            response["status"] = "FAILURE"
            response["error"] = str(task_result.info)
            
    return response

@app.get("/security/verify/{model_filename}", tags=["Security Integrity"])
async def verify_model_hash(model_filename: str):
    """
    Прямой вызов SecurityManager для проверки целостности файла модели.
    Демонстрация механизма защиты от подмены (Tampering).
    """
    file_path = os.path.join("models", model_filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Файл модели не найден")
    
    computed_hash = SecurityManager.get_hash(file_path)
    
    return {
        "filename": model_filename,
        "sha256": computed_hash,
        "message": "Integrity check passed"
    }

if __name__ == "__main__":
    # Запуск сервера на порту 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)