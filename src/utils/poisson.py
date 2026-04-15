"""
Local Poisson Density Estimation (§3)

Estimates local density λ(x) and local restriction fraction α(q)
using the k-NN distance-based estimator.
"""

import numpy as np
from scipy.special import gammaln


def estimate_local_density(
    query: np.ndarray,
    database: np.ndarray,
    k: int = 10,
) -> float:
    """Estimate local Poisson density λ(q) from k-NN distance."""
    d = query.shape[0]
    distances = np.linalg.norm(database - query, axis=1)
    d_k = np.partition(distances, k)[k]

    c_d = np.pi ** (d / 2) / np.exp(gammaln(d / 2 + 1))
    V_k = c_d * d_k ** d
    return k / V_k


def estimate_local_alpha(
    query: np.ndarray,
    authorized: np.ndarray,
    full_database: np.ndarray,
    k: int = 10,
) -> float:
    """Estimate local restriction fraction α(q) = λ̄_u(q) / λ(q)."""
    lambda_full = estimate_local_density(query, full_database, k)
    lambda_auth = estimate_local_density(query, authorized, k)
    if lambda_full < 1e-10:
        return 0.0
    return max(0.0, 1.0 - lambda_auth / lambda_full)
