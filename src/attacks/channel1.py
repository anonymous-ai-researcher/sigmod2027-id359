"""
Channel 1: Distance Distribution Skew Attack (§3.1)

Detects the presence of restricted vectors by comparing the observed k-NN
distance distribution against the expected distribution under the visible-only
hypothesis.

Uses a Gamma likelihood-ratio test on the k-NN volume V_k = c_d · d_k^d.

Key result: KL(Q || P) = k[α/(1-α) + ln(1-α)], scaling linearly in k
and quadratically in α (Proposition 3.1).

Reference: §3.1, Proposition 3.1
"""

import numpy as np
from scipy import stats
from scipy.special import gammaln


def distance_skew_score(
    distances: np.ndarray,
    k: int,
    d: int,
    lambda_estimate: float,
) -> float:
    """
    Compute the distance skew attack score for a single query.

    Parameters
    ----------
    distances : np.ndarray, shape (k,)
        Distances to k nearest neighbors from access-controlled search.
    k : int
        Number of neighbors.
    d : int
        Embedding dimension.
    lambda_estimate : float
        Estimated full-database density λ (from public data statistics).

    Returns
    -------
    float
        Log-likelihood ratio score (higher = more evidence of restricted vectors).
    """
    # Volume of the k-th NN ball: V_k = c_d · d_k^d
    c_d = np.pi ** (d / 2) / np.exp(gammaln(d / 2 + 1))
    d_k = distances[-1]  # k-th nearest distance
    V_k = c_d * d_k ** d

    # Under H0 (no restriction): V_k ~ Gamma(k, 1/λ)
    # Under H1 (restriction): V_k ~ Gamma(k, 1/λ_u) with λ_u < λ
    # LLR = log p(V_k | H1) - log p(V_k | H0)
    theta_null = 1.0 / lambda_estimate  # full database
    # Estimate λ_u from the observed distances
    theta_alt = V_k / k  # MLE from single observation

    if theta_alt <= theta_null:
        return 0.0  # No evidence of restriction

    llr = (
        stats.gamma.logpdf(V_k, a=k, scale=theta_alt) -
        stats.gamma.logpdf(V_k, a=k, scale=theta_null)
    )
    return float(llr)


def evaluate_channel1(
    queries: np.ndarray,
    all_distances: list[np.ndarray],
    labels: np.ndarray,
    k: int,
    d: int,
    lambda_estimate: float,
) -> float:
    """
    Evaluate Channel 1 attack AUC over a set of queries.

    Parameters
    ----------
    queries : np.ndarray, shape (n_queries, d)
    all_distances : list of np.ndarray, each shape (k,)
        k-NN distances for each query.
    labels : np.ndarray, shape (n_queries,), binary
        1 = restricted vectors present near query, 0 = not present.
    k : int
    d : int
    lambda_estimate : float

    Returns
    -------
    float
        AUC of the attack classifier.
    """
    from sklearn.metrics import roc_auc_score

    scores = []
    for dists in all_distances:
        score = distance_skew_score(dists, k, d, lambda_estimate)
        scores.append(score)

    return roc_auc_score(labels, scores)
