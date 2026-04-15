"""
Private Top-k Selection via Gumbel Noise (§4.1)

Replaces the deterministic k-NN operator with a randomized mechanism that adds
calibrated Gumbel noise to relevance scores, then selects top-k by noisy scores.

Key properties:
- ε-differential privacy with respect to addition/removal of a single vector (Theorem 1)
- Zero sensitivity of distance-based scores enables tight noise calibration
- One-shot selection avoids k-fold composition degradation

Reference: Definition 4.1, Theorem 1, Proposition 4.6
"""

import numpy as np
from typing import Optional


def gumbel_noise(shape: tuple, scale: float, rng: np.random.Generator) -> np.ndarray:
    """Sample Gumbel(0, scale) noise."""
    u = rng.uniform(1e-10, 1.0 - 1e-10, size=shape)
    return -scale * np.log(-np.log(u))


def private_topk(
    query: np.ndarray,
    candidates: np.ndarray,
    epsilon: float,
    k: int,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Private top-k selection via the joint exponential mechanism (Gumbel noise).

    Parameters
    ----------
    query : np.ndarray, shape (d,)
        Query vector.
    candidates : np.ndarray, shape (n, d)
        Candidate vectors from the authorized set D_u (or augmented D_u^+).
    epsilon : float
        Per-query privacy budget ε(q).
    k : int
        Number of neighbors to return.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    indices : np.ndarray, shape (k,)
        Indices of selected neighbors in the candidate array.
    distances : np.ndarray, shape (k,)
        Distances from query to selected neighbors.

    Notes
    -----
    The score function s(q, v) = -||q - v||_2 has zero sensitivity with respect
    to the restricted set D̄_u (Theorem 1), so the Gumbel noise is calibrated
    purely to mask the presence/absence of candidates.

    The Gumbel scale is 2/ε, following the joint exponential mechanism
    (Gillenwater et al., ICML 2022).
    """
    if rng is None:
        rng = np.random.default_rng()

    # Compute relevance scores: s(q, v) = -dist(q, v)
    distances = np.linalg.norm(candidates - query, axis=1)
    scores = -distances

    # Add Gumbel noise with scale 2/ε
    noise_scale = 2.0 / epsilon
    noisy_scores = scores + gumbel_noise(scores.shape, noise_scale, rng)

    # Select top-k by noisy scores
    topk_indices = np.argpartition(noisy_scores, -k)[-k:]
    topk_indices = topk_indices[np.argsort(noisy_scores[topk_indices])[::-1]]

    return topk_indices, distances[topk_indices]


def recall_at_k(
    selected_indices: np.ndarray,
    true_topk_indices: np.ndarray,
) -> float:
    """
    Compute Recall@k = |R ∩ R*| / k.

    Parameters
    ----------
    selected_indices : indices returned by private_topk
    true_topk_indices : indices of the true top-k (from exact k-NN)
    """
    k = len(true_topk_indices)
    overlap = len(set(selected_indices) & set(true_topk_indices))
    return overlap / k


def expected_recall_bound(
    n_u: int, k: int, epsilon: float, gamma_k: float
) -> float:
    """
    Theoretical lower bound on E[Recall@k] (Proposition 4.6).

    E[Recall@k] >= 1 - (n_u - k) * exp(-ε * γ_k / 2)

    Parameters
    ----------
    n_u : int
        Size of authorized set.
    k : int
        Number of neighbors.
    epsilon : float
        Privacy budget.
    gamma_k : float
        Score gap: s(q, v_(k)) - s(q, v_(k+1)).
    """
    return 1.0 - (n_u - k) * np.exp(-epsilon * gamma_k / 2.0)
