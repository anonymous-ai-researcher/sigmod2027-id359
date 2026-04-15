"""
Decoy-Augmented Indexing (§4.3)

Generates synthetic decoy vectors to fill geometric gaps left by restricted
vectors, providing plausible deniability against the topology distortion
channel (Channel 2).

Two generation modes:
- Isotropic: Gaussian perturbation in all d dimensions (Eq. 5)
- Manifold-aware: perturbation projected onto local tangent space (Eq. 6)

Key results:
- Proposition 4.10: No score-noise mechanism can close Channel 2
- Proposition 4.7: Manifold-aware decoys are d/d_int times more efficient
- Proposition 4.8: c=1 decoy reduces Rayleigh statistic by factor 1-α

Reference: §4.3, Eqs. 5-6, Propositions 4.7-4.8, 4.10
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from typing import Optional


def generate_isotropic_decoys(
    restricted_vectors: np.ndarray,
    sigma_dec: float,
    c: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate isotropic decoy vectors (Eq. 5).

    v*_dec,i = v* + ξ_i,  ξ_i ~ N(0, σ_dec² I_d)

    Parameters
    ----------
    restricted_vectors : np.ndarray, shape (n_restricted, d)
        Restricted vectors to generate decoys for.
    sigma_dec : float
        Perturbation scale. Calibrated to median distance from restricted
        vectors to their 10-th nearest authorized neighbor.
    c : int
        Number of decoys per restricted vector.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    decoys : np.ndarray, shape (n_restricted * c, d)
        Generated decoy vectors.
    """
    if rng is None:
        rng = np.random.default_rng()

    n, d = restricted_vectors.shape
    decoys = []

    for v_star in restricted_vectors:
        for _ in range(c):
            xi = rng.normal(0, sigma_dec, size=d)
            decoy = v_star + xi
            decoys.append(decoy)

    return np.array(decoys)


def generate_manifold_decoys(
    restricted_vectors: np.ndarray,
    authorized_vectors: np.ndarray,
    sigma_man: float,
    c: int = 1,
    d_int: Optional[int] = None,
    pca_neighbors: int = 50,
    variance_threshold: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate manifold-aware decoy vectors (Eq. 6).

    v*_man,i = v* + U_{v*} ζ_i,  ζ_i ~ N(0, σ_man² I_{d_int})

    where U_{v*} is the local PCA basis (top d_int principal components
    from the ℓ nearest neighbors of v*).

    Parameters
    ----------
    restricted_vectors : np.ndarray, shape (n_restricted, d)
    authorized_vectors : np.ndarray, shape (n_authorized, d)
        Used for computing local PCA bases.
    sigma_man : float
        Perturbation scale in tangent space.
    c : int
        Number of decoys per restricted vector.
    d_int : int, optional
        Intrinsic dimensionality. If None, estimated via PCA.
    pca_neighbors : int
        Number of neighbors (ℓ) for local PCA.
    variance_threshold : float
        Cumulative variance threshold for d_int estimation.
    rng : np.random.Generator, optional

    Returns
    -------
    decoys : np.ndarray, shape (n_restricted * c, d)
    """
    if rng is None:
        rng = np.random.default_rng()

    n_restricted, d = restricted_vectors.shape

    # Build neighbor index on authorized vectors
    nn = NearestNeighbors(n_neighbors=pca_neighbors, metric="euclidean")
    nn.fit(authorized_vectors)

    decoys = []
    for v_star in restricted_vectors:
        # Find ℓ nearest authorized neighbors
        _, neighbor_idx = nn.kneighbors(v_star.reshape(1, -1))
        neighbors = authorized_vectors[neighbor_idx[0]]

        # Local PCA to estimate tangent space
        pca = PCA()
        pca.fit(neighbors)

        if d_int is None:
            # Estimate d_int from cumulative variance
            cumvar = np.cumsum(pca.explained_variance_ratio_)
            local_d_int = int(np.searchsorted(cumvar, variance_threshold) + 1)
            local_d_int = min(local_d_int, pca_neighbors - 1, d)
        else:
            local_d_int = d_int

        # U_{v*}: top d_int principal components
        U = pca.components_[:local_d_int].T  # shape (d, d_int)

        for _ in range(c):
            # ζ ~ N(0, σ_man² I_{d_int})
            zeta = rng.normal(0, sigma_man, size=local_d_int)
            # Project onto tangent space
            decoy = v_star + U @ zeta
            decoys.append(decoy)

    return np.array(decoys)


def calibrate_sigma(
    restricted_vectors: np.ndarray,
    authorized_vectors: np.ndarray,
    k_ref: int = 10,
) -> float:
    """
    Calibrate σ_dec to median distance from restricted vectors to their
    k_ref-th nearest authorized neighbor.

    Reference: Appendix D, Hyperparameter Table
    """
    nn = NearestNeighbors(n_neighbors=k_ref, metric="euclidean")
    nn.fit(authorized_vectors)
    distances, _ = nn.kneighbors(restricted_vectors)
    return float(np.median(distances[:, -1]))


def estimate_intrinsic_dim(
    vectors: np.ndarray,
    variance_threshold: float = 0.95,
) -> int:
    """
    Estimate intrinsic dimensionality via PCA (95% cumulative variance).

    Reference: §6.4, Appendix D
    """
    pca = PCA()
    pca.fit(vectors)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    d_int = int(np.searchsorted(cumvar, variance_threshold) + 1)
    return d_int
