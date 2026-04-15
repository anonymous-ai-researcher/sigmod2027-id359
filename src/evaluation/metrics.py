"""Evaluation metrics: AUC, Recall@k, NDCG@k (§6.1)."""

from sklearn.metrics import roc_auc_score
import numpy as np


def compute_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute AUC of attack classifier. AUC = 0.5 means perfect defense."""
    return roc_auc_score(labels, scores)


def excess_auc_reduction(auc_before: float, auc_after: float) -> float:
    """
    Compute percentage reduction in excess AUC above random (0.5).
    Used for the '80-90% reduction' claim in §6.
    """
    excess_before = auc_before - 0.5
    excess_after = auc_after - 0.5
    if excess_before <= 0:
        return 0.0
    return (excess_before - excess_after) / excess_before
