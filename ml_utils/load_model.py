import os
from catboost import CatBoostClassifier


def get_model_path(path: str) -> str:
    """Получение пути модели в зависимости от того выполняется код локально или на ЛМС"""
    if os.environ.get("IS_LMS") == "1":
        model_path = '/workdir/user_input/model'
    else:
        model_path = path
    return model_path


def load_models() -> CatBoostClassifier:
    """Загрузка модели"""
    model_path = get_model_path("ml_utils/catboost_model")
    mod = CatBoostClassifier()
    mod.load_model(model_path)
    return mod
