from typing import Dict
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from base import FactorizationModel


def get_rating(user_id: int, item_id: int, model: FactorizationModel) -> float:
    return model.get_user_emb(user_id) @ model.get_item_emb(item_id)


def get_similar_items(item_id: int, model: FactorizationModel, top_k: int = 10) -> np.ndarray:
    _, I = model.factorize()
    scores = cosine_similarity(I)[item_id]
    return np.argsort(-scores)[:top_k]


def get_recommendations(user_id: int, model: FactorizationModel, top_k: int = 10) -> np.ndarray:
    _, I = model.factorize()
    scores = I @ model.get_user_emb(user_id)
    return np.argsort(-scores)[:top_k]


def log_iter(iteration: int, metrics: Dict[str, float], elapsed_time: float):
    iter_out = f"Iter: {iteration}"
    metrics_out = [f"{key}: {value:.2f}" for key, value in metrics.items()]
    time_out = f"Elapsed: {int(elapsed_time // 60)}m{int(elapsed_time % 60)}s"
    print(" | ".join([iter_out] + metrics_out + [time_out]))


def rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return np.sqrt(((y_true - y_pred) ** 2).mean())
