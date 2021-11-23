import numpy as np
from sklearn.metrics import roc_auc_score


def dcg(scores: np.ndarray, relevance: np.ndarray) -> float:
    assert len(scores) == len(relevance)

    n_items = len(scores)
    sorted_idx = np.argsort(-scores)
    relevance = relevance[sorted_idx]

    return sum(relevance / np.log2(1 + np.arange(1, n_items + 1)))


def idcg(scores: np.ndarray, relevance: np.ndarray) -> float:
    assert len(scores) == len(relevance)

    n_items = len(scores)
    sorted_idx = np.argsort(-relevance)
    relevance = relevance[sorted_idx]

    return sum(relevance / np.log2(1 + np.arange(1, n_items + 1)))


def ndcg(queries: np.ndarray, scores: np.ndarray, relevance: np.ndarray) -> float:
    assert len(queries) == len(scores) == len(relevance)

    query_labels = np.unique(queries)
    ndcgs = []

    for query_label in query_labels:
        query_mask = queries == query_label
        query_dcg = dcg(scores[query_mask], relevance[query_mask])
        query_idcg = idcg(scores[query_mask], relevance[query_mask])

        if query_idcg > 0:
            ndcgs.append(query_dcg / query_idcg)

    return np.array(ndcgs).mean()


def auc_per_query(queries: np.ndarray, scores: np.ndarray, relevance: np.ndarray) -> float:
    assert len(queries) == len(scores) == len(relevance)

    query_labels = np.unique(queries)
    aucs = []

    for query_label in query_labels:
        try:
            query_mask = queries == query_label
            query_auc = roc_auc_score(relevance[query_mask], scores[query_mask])
            aucs.append(query_auc)
        except:
            pass

    return np.array(aucs).mean()
