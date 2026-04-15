"""
Channel 2: Neighborhood Topology Distortion Attack (§3.2)

Detects directional asymmetry in k-NN results using the Rayleigh test.
When restricted vectors occupy a particular direction from the query,
authorized neighbors shift away, producing a detectable angular gap.

Key result: Rayleigh test statistic T = kd||μ||² has power 1-β when
k ≥ χ²_{d,1-β} / (d · α² · g(Ω,d)²) (Proposition 3.2).

Reference: §3.2, Proposition 3.2
"""

import numpy as np
from scipy import stats


def compute_rayleigh_statistic(
    query: np.ndarray,
    neighbors: np.ndarray,
    d_eff: int = None,
) -> float:
    """
    Compute Rayleigh test statistic T = k·d·||μ||².

    Parameters
    ----------
    query : np.ndarray, shape (d,)
    neighbors : np.ndarray, shape (k, d)
    d_eff : int, optional
        Effective dimensionality (d_int). If None, use ambient d.

    Returns
    -------
    float
        Rayleigh test statistic T.
    """
    k, d = neighbors.shape
    if d_eff is None:
        d_eff = d

    # Unit directions ω_j = (v_(j) - q) / ||v_(j) - q||
    diffs = neighbors - query
    norms = np.linalg.norm(diffs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    directions = diffs / norms

    # Mean direction μ = (1/k) Σ ω_j
    mu = directions.mean(axis=0)

    # Rayleigh statistic T = k · d_eff · ||μ||²
    T = k * d_eff * np.sum(mu ** 2)
    return T


def topology_attack_score(
    query: np.ndarray,
    neighbors: np.ndarray,
    d_eff: int = None,
) -> float:
    """
    Compute topology attack score (p-value of Rayleigh test).

    Under H0 (no restriction, isotropic data), T ~ χ²_d (approximately).
    Lower p-value = more evidence of directional asymmetry.

    Returns negative log p-value as the attack score (higher = more leakage).
    """
    k, d = neighbors.shape
    if d_eff is None:
        d_eff = d

    T = compute_rayleigh_statistic(query, neighbors, d_eff)
    p_value = 1.0 - stats.chi2.cdf(T, df=d_eff)
    return -np.log(max(p_value, 1e-300))


def evaluate_channel2(
    queries: np.ndarray,
    all_neighbors: list[np.ndarray],
    labels: np.ndarray,
    d_int: int = None,
) -> float:
    """
    Evaluate Channel 2 attack AUC.

    Parameters
    ----------
    queries : np.ndarray, shape (n_queries, d)
    all_neighbors : list of np.ndarray, each shape (k, d)
    labels : np.ndarray, shape (n_queries,), binary
    d_int : int, optional
        Intrinsic dimensionality for the Rayleigh test.
    """
    from sklearn.metrics import roc_auc_score

    scores = []
    for q, neighbors in zip(queries, all_neighbors):
        score = topology_attack_score(q, neighbors, d_eff=d_int)
        scores.append(score)

    return roc_auc_score(labels, scores)
