"""
Geometry-Aware Budget Allocation (§4.2)

Adapts the effective privacy budget ε(q) to local leakage risk ρ(q),
allocating less noise in low-risk regions and more noise where restricted
vectors are concentrated.

Key result: The 1/ρ(q) allocation is provably optimal under a bi-constrained
privacy-utility formulation (Proposition 4.9, proved via KKT conditions).

Reference: Eq. 7-8, Proposition 4.9
"""

import numpy as np
from typing import Optional


def compute_leakage_risk(
    alpha_q: float,
    k: int,
) -> float:
    """
    Compute leakage risk at query point q (Eq. 8).

    ρ(q) = α(q) · √k

    Parameters
    ----------
    alpha_q : float
        Local restriction fraction α(q) = λ̄_u(q) / λ(q).
    k : int
        Number of neighbors.
    """
    return alpha_q * np.sqrt(k)


def adaptive_epsilon(
    rho_q: float,
    epsilon_0: float,
    rho_max: float,
    rho_min: float,
) -> float:
    """
    Compute per-query privacy budget ε(q) (Eq. 7).

    ε(q) = ε_0 · ρ_max / max(ρ(q), ρ_min)

    Parameters
    ----------
    rho_q : float
        Leakage risk at query point.
    epsilon_0 : float
        Global base privacy budget.
    rho_max : float
        Maximum leakage risk (95th percentile over training queries).
    rho_min : float
        Floor on leakage risk (controls max effective budget).

    Returns
    -------
    float
        Per-query privacy budget ε(q).

    Notes
    -----
    - ρ(q) ≈ ρ_max → ε(q) ≈ ε_0 (high risk, strong protection)
    - ρ(q) ≈ ρ_min → ε(q) = ε_0 · ρ_max/ρ_min (low risk, relaxed noise)
    - ρ_min = ρ_max recovers uniform allocation
    """
    return epsilon_0 * rho_max / max(rho_q, rho_min)


class RiskMap:
    """
    Differentially private risk map for offline ρ(q) estimation.

    The risk map is constructed during index build time with access to the
    full database D and policy π. It releases a grid-based density estimate
    under (ε_ρ, δ_ρ)-DP.

    Reference: §4.2 "Computing ρ(q) without leaking"
    """

    def __init__(
        self,
        epsilon_rho: float = 0.1,
        delta_rho: float = 1e-6,
        min_vectors_per_cell: int = 100,
    ):
        self.epsilon_rho = epsilon_rho
        self.delta_rho = delta_rho
        self.min_vectors_per_cell = min_vectors_per_cell
        self.grid = None
        self.rho_values = None

    def build(
        self,
        database: np.ndarray,
        restricted_mask: np.ndarray,
        k: int,
    ) -> None:
        """
        Build the risk map from the full database.

        Parameters
        ----------
        database : np.ndarray, shape (n, d)
            Full database vectors.
        restricted_mask : np.ndarray, shape (n,), bool
            True for restricted vectors.
        k : int
            Number of neighbors.
        """
        n, d = database.shape

        # Adaptive grid: cell side chosen so each cell has >= min_vectors_per_cell
        n_cells_per_dim = max(1, int((n / self.min_vectors_per_cell) ** (1.0 / d)))
        # In practice, use a much coarser grid (project to low-dim first)
        # Here we use a simplified grid on the first 2 principal components
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(2, d))
        coords = pca.fit_transform(database)

        n_bins = max(1, int(np.sqrt(n / self.min_vectors_per_cell)))
        self.n_bins = n_bins

        # Compute per-cell restriction fraction
        x_edges = np.linspace(coords[:, 0].min(), coords[:, 0].max(), n_bins + 1)
        y_edges = np.linspace(coords[:, 1].min(), coords[:, 1].max(), n_bins + 1) if coords.shape[1] > 1 else np.array([0, 1])

        self.x_edges = x_edges
        self.y_edges = y_edges
        self.pca = pca

        rho_grid = np.zeros((n_bins, max(1, n_bins)))
        for i in range(n_bins):
            for j in range(max(1, n_bins)):
                if coords.shape[1] > 1:
                    mask = (
                        (coords[:, 0] >= x_edges[i]) & (coords[:, 0] < x_edges[i + 1]) &
                        (coords[:, 1] >= y_edges[j]) & (coords[:, 1] < y_edges[j + 1])
                    )
                else:
                    mask = (coords[:, 0] >= x_edges[i]) & (coords[:, 0] < x_edges[i + 1])

                n_cell = mask.sum()
                n_restricted = (mask & restricted_mask).sum()

                if n_cell > 0:
                    alpha_cell = n_restricted / n_cell
                else:
                    alpha_cell = 0.0

                # Add Laplace noise for DP release
                sensitivity = 1.0 / max(n_cell, 1)
                noise = np.random.laplace(0, sensitivity / self.epsilon_rho)
                alpha_noisy = np.clip(alpha_cell + noise, 0.0, 1.0)

                rho_grid[i, j] = alpha_noisy * np.sqrt(k)

        self.rho_values = rho_grid
        self.rho_max = np.percentile(rho_grid[rho_grid > 0], 95) if (rho_grid > 0).any() else 1.0

    def query(self, q: np.ndarray) -> float:
        """Look up ρ(q) from the risk map."""
        coords = self.pca.transform(q.reshape(1, -1))[0]
        i = np.searchsorted(self.x_edges[1:], coords[0])
        i = min(i, self.rho_values.shape[0] - 1)
        if self.rho_values.ndim > 1 and len(coords) > 1:
            j = np.searchsorted(self.y_edges[1:], coords[1])
            j = min(j, self.rho_values.shape[1] - 1)
            return self.rho_values[i, j]
        return self.rho_values[i, 0]
