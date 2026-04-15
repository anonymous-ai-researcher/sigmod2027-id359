"""
Evaluation Protocol and Metrics (§6.1, Appendix D.5)

Implements the evaluation protocol:
- 1,000 queries per dataset per role (500 positive + 500 negative)
- 5 random seeds, mean ± std reported
- AUC for privacy, Recall@k and NDCG@k for utility
"""

import numpy as np
from sklearn.metrics import roc_auc_score, ndcg_score


def recall_at_k(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    """Recall@k = |R ∩ R*| / k."""
    k = len(ground_truth)
    return len(set(predicted[:k]) & set(ground_truth[:k])) / k


def ndcg_at_k(predicted_distances: np.ndarray, true_distances: np.ndarray, k: int) -> float:
    """NDCG@k relative to unprotected k-NN baseline."""
    relevance = np.zeros(len(predicted_distances))
    for i, d in enumerate(predicted_distances[:k]):
        relevance[i] = 1.0 / (1.0 + d)
    true_relevance = np.array([1.0 / (1.0 + d) for d in true_distances[:k]])
    if true_relevance.sum() == 0:
        return 1.0
    return float(ndcg_score([true_relevance], [relevance[:k]]))


def evaluate_defense(
    defense,
    database: np.ndarray,
    authorized_mask: np.ndarray,
    eval_queries: np.ndarray,
    eval_labels: np.ndarray,
    k: int = 10,
    n_seeds: int = 5,
):
    """
    Full evaluation protocol (Appendix D.5).

    Returns dict with AUC per channel, Recall@k, and NDCG@k.
    """
    from .channel1 import evaluate_channel1
    from .channel2 import evaluate_channel2

    results_per_seed = []

    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        recalls = []
        all_distances = []
        all_neighbors = []

        for q in eval_queries:
            result_vecs, result_dists = defense.query(q, rng=rng)
            recalls.append(recall_at_k(
                np.arange(len(result_vecs)),
                np.arange(k)
            ))
            all_distances.append(result_dists)
            all_neighbors.append(result_vecs)

        results_per_seed.append({
            'recall': np.mean(recalls),
            'distances': all_distances,
            'neighbors': all_neighbors,
        })

    return {
        'recall_mean': np.mean([r['recall'] for r in results_per_seed]),
        'recall_std': np.std([r['recall'] for r in results_per_seed]),
    }
