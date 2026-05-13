import time
import logging
from tasks import train_fair_model_task, run_audit_task

# Настройка логгера
logging.basicConfig(level=logging.INFO, format='%(asctime)s | EXPERIMENT | %(message)s')
logger = logging.getLogger(__name__)

def run_experiment_pipeline():
    """
    Автоматизированный сценарий A/B тестирования.
    Последовательно запускает обучение и аудит модели.
    """
    logger.info("--- ЗАПУСК ЭКСПЕРИМЕНТАЛЬНОГО ПАЙПЛАЙНА ---")
    
    # 1. КОНФИГУРАЦИЯ ЭКСПЕРИМЕНТА
    # Параметры соответствуют Главе 5 диссертации
    DATA_PATH = "data/adult_processed.csv"
    TEST_DATA_PATH = "data/adult_processed.csv"  

    
    config = {
        "features": ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week'],
        "target": "target",
        "protected": "sex_binary",
        "lambda_reg": 15.0, # Оптимальное значение (Elbow Point)
        "epochs": 30,
        "batch_size": 1024
    }
    
    # 2. ОБУЧЕНИЕ МОДЕЛИ (Training Phase)
    logger.info(f"Запуск обучения с lambda={config['lambda_reg']}...")
    
    # Используем .apply() для синхронного ожидания результата в скрипте
    # (в реальном API используется .delay)
    try:
        train_result = train_fair_model_task.apply(args=[DATA_PATH, config]).get()
    except Exception as e:
        logger.error(f"Ошибка обучения: {e}")
        return

    if train_result["status"] != "COMPLETED":
        logger.error("Обучение завершилось неудачей.")
        return

    model_path = train_result["model_path"]
    model_hash = train_result["model_hash"]
    logger.info(f"Модель успешно обучена.")
    logger.info(f"Путь: {model_path}")
    logger.info(f"SHA-256 Hash: {model_hash}")

    # 3. АУДИТ МОДЕЛИ (Audit Phase)
    logger.info("Запуск процедуры аудита...")
    
    audit_config = {
        "features": config["features"],
        "target": config["target"],
        "protected": config["protected"],
        "expected_hash": model_hash # Проверка целостности перед аудитом
    }
    
    try:
        audit_result = run_audit_task.apply(args=[model_path, TEST_DATA_PATH, audit_config]).get()
    except Exception as e:
        logger.error(f"Ошибка аудита: {e}")
        return
        
    # 4. ВЫВОД РЕЗУЛЬТАТОВ
    metrics = audit_result["metrics"]
    logger.info("-" * 30)
    logger.info("ИТОГОВЫЙ ОТЧЕТ:")
    logger.info(f"Accuracy: {metrics.get('accuracy', 0.0):.4f}")
    logger.info(f"F1-Score: {metrics.get('f1_score', 0.0):.4f}")
    logger.info(f"Disparate Impact: {metrics.get('disparate_impact', 0.0):.4f}")
    logger.info(f"SPD: {metrics.get('statistical_parity_difference', 0.0):.4f}")
    logger.info("-" * 30)
    
    # Автоматическая проверка критериев успеха (Quality Gate)
    if metrics['disparate_impact'] >= 0.8:
        logger.info("✅ SUCCESS: Модель соответствует требованиям ГОСТ (DI > 0.8).")
    else:
        logger.warning("❌ FAILURE: Обнаружена дискриминация.")

if __name__ == "__main__":
    run_experiment_pipeline()
