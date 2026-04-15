"""
Composed Defense Mechanism (§4.4, Algorithm 1)

Integrates private top-k selection, geometry-aware budget allocation,
and decoy-augmented indexing into a complete query-processing pipeline.

The composed mechanism satisfies (ε, δ)-DP via Rényi composition (Theorem 2).

Reference: Algorithm 1, Theorem 2, §5 System Architecture
"""

import numpy as np
from typing import Optional
from .private_topk import private_topk
from .geometry_aware import RiskMap, adaptive_epsilon, compute_leakage_risk
from .decoy import (
    generate_isotropic_decoys, generate_manifold_decoys,
    calibrate_sigma, estimate_intrinsic_dim,
)


class ComposedDefense:
    """
    Access-controlled private k-NN query system (Algorithm 1).

    Composes three defense components:
    1. Private top-k via Gumbel noise (§4.1)
    2. Geometry-aware budget allocation (§4.2)
    3. Decoy-augmented indexing (§4.3)
    """

    def __init__(
        self,
        epsilon_0: float = 1.0,
        delta: float = 1e-6,
        k: int = 10,
        k_prime_factor: int = 2,
        c: int = 1,
        rho_min_ratio: float = 0.1,
        manifold_aware: bool = True,
        d_int: Optional[int] = None,
        pca_neighbors: int = 50,
    ):
        self.epsilon_0 = epsilon_0
        self.delta = delta
        self.k = k
        self.k_prime = k_prime_factor * k
        self.c = c
        self.rho_min_ratio = rho_min_ratio
        self.manifold_aware = manifold_aware
        self.d_int = d_int
        self.pca_neighbors = pca_neighbors

        self.risk_map = None
        self.augmented_index = None
        self.decoy_flags = None
        self.rho_max = None
        self.rho_min = None
        self.query_count = 0
        self.total_rdp_cost = 0.0

    def build(
        self,
        database: np.ndarray,
        restricted_mask: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """
        Offline preprocessing: build risk map, generate decoys, build index.

        Parameters
        ----------
        database : np.ndarray, shape (n, d)
        restricted_mask : np.ndarray, shape (n,), bool
        rng : np.random.Generator, optional
        """
        if rng is None:
            rng = np.random.default_rng()

        authorized = database[~restricted_mask]
        restricted = database[restricted_mask]

        # 1. Build risk map (§4.2)
        self.risk_map = RiskMap()
        self.risk_map.build(database, restricted_mask, self.k)
        self.rho_max = self.risk_map.rho_max
        self.rho_min = self.rho_max * self.rho_min_ratio

        # 2. Estimate intrinsic dimensionality
        if self.d_int is None:
            self.d_int = estimate_intrinsic_dim(authorized)

        # 3. Calibrate decoy noise scale
        sigma = calibrate_sigma(restricted, authorized, k_ref=10)

        # 4. Generate decoys (§4.3)
        if self.c > 0 and len(restricted) > 0:
            if self.manifold_aware:
                decoys = generate_manifold_decoys(
                    restricted, authorized, sigma,
                    c=self.c, d_int=self.d_int,
                    pca_neighbors=self.pca_neighbors, rng=rng,
                )
            else:
                decoys = generate_isotropic_decoys(
                    restricted, sigma, c=self.c, rng=rng,
                )

            # Build augmented index D_u^+ = D_u ∪ D_dec
            self.augmented_vectors = np.vstack([authorized, decoys])
            self.decoy_flags = np.concatenate([
                np.zeros(len(authorized), dtype=bool),
                np.ones(len(decoys), dtype=bool),
            ])
        else:
            self.augmented_vectors = authorized
            self.decoy_flags = np.zeros(len(authorized), dtype=bool)

    def query(
        self,
        q: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Process a single private k-NN query (Algorithm 1).

        Parameters
        ----------
        q : np.ndarray, shape (d,)
            Query vector.
        rng : np.random.Generator, optional

        Returns
        -------
        result_vectors : np.ndarray, shape (k, d)
            Selected neighbor vectors (genuine only, decoys filtered).
        result_distances : np.ndarray, shape (k,)
            Distances to selected neighbors.
        """
        if rng is None:
            rng = np.random.default_rng()

        # Step 1: Risk lookup → ε(q)
        rho_q = self.risk_map.query(q)
        eps_q = adaptive_epsilon(rho_q, self.epsilon_0, self.rho_max, self.rho_min)

        # Step 2: ANN search on D_u^+ with k' = 2k candidates
        # (In production, use HNSW/IVF-PQ; here we use exact search for correctness)
        distances = np.linalg.norm(self.augmented_vectors - q, axis=1)
        candidate_indices = np.argpartition(distances, self.k_prime)[:self.k_prime]

        # Step 3: Private top-k with Gumbel noise
        candidates = self.augmented_vectors[candidate_indices]
        selected_local, _ = private_topk(q, candidates, eps_q, self.k_prime, rng)
        selected_global = candidate_indices[selected_local]

        # Step 4: Filter decoys — select first k genuine vectors
        genuine_mask = ~self.decoy_flags[selected_global]
        genuine_indices = selected_global[genuine_mask][:self.k]

        # Pad if fewer than k genuine vectors survived
        if len(genuine_indices) < self.k:
            remaining = selected_global[~np.isin(selected_global, genuine_indices)]
            remaining_genuine = remaining[~self.decoy_flags[remaining]]
            genuine_indices = np.concatenate([genuine_indices, remaining_genuine])[:self.k]

        # Step 5: Update RDP accounting
        self.query_count += 1
        rdp_cost = eps_q * (np.exp(eps_q) - 1) / 2.0
        self.total_rdp_cost += rdp_cost

        result_vectors = self.augmented_vectors[genuine_indices]
        result_distances = np.linalg.norm(result_vectors - q, axis=1)
        sort_order = np.argsort(result_distances)

        return result_vectors[sort_order], result_distances[sort_order]

    def get_total_epsilon(self, delta: Optional[float] = None, lam: float = 3.0) -> float:
        """
        Compute total (ε, δ)-DP budget via RDP-to-DP conversion (Theorem 2, Eq. 9).

        ε_total = total_rdp_cost + ln(1/δ) / (λ - 1)
        """
        if delta is None:
            delta = self.delta
        return self.total_rdp_cost + np.log(1.0 / delta) / (lam - 1)

    def get_session_limit(self, epsilon_total_max: float = 10.0) -> int:
        """
        Compute maximum queries before exceeding ε_total budget.

        m_max ≈ ε_total_max / (ε̄_q · (e^{ε̄_q} - 1) / 2)
        """
        eps_bar = self.epsilon_0 * self.rho_max / self.rho_min
        per_query_rdp = eps_bar * (np.exp(eps_bar) - 1) / 2.0
        return int(epsilon_total_max / per_query_rdp)
