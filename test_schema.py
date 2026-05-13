from schemas import TrainingRequest
try:
    # Пример невалидного запроса (отрицательная лямбда)
    req = TrainingRequest(dataset_path="data.csv", features=["age"], target="y", protected="z", lambda_reg=-1.0)
except Exception as e:
    print(f"Валидация сработала: {e}")
