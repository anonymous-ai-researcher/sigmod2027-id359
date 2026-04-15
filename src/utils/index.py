"""
ANN Index Wrapper and Access Control (§2.1, §5)

Provides unified interface for HNSW and IVF-PQ indexes with
access control policy management.
"""

import numpy as np
from typing import Optional


class AccessControlPolicy:
    """
    Access control policy π: U × D → {0, 1} (Definition 2).

    Supports RBAC and ABAC through role-based or attribute-based
    permission assignments.
    """

    def __init__(self, n_vectors: int, n_roles: int):
        self.n_vectors = n_vectors
        self.n_roles = n_roles
        self.permissions = np.ones((n_roles, n_vectors), dtype=bool)

    def set_restricted(self, role_id: int, vector_indices: np.ndarray):
        """Mark vectors as restricted for a given role."""
        self.permissions[role_id, vector_indices] = False

    def get_authorized_mask(self, role_id: int) -> np.ndarray:
        """Get boolean mask of authorized vectors for a role."""
        return self.permissions[role_id]

    def get_restricted_mask(self, role_id: int) -> np.ndarray:
        """Get boolean mask of restricted vectors for a role."""
        return ~self.permissions[role_id]

    def get_alpha(self, role_id: int) -> float:
        """Get restriction ratio α_u = |D̄_u| / |D|."""
        return (~self.permissions[role_id]).sum() / self.n_vectors


class ANNIndex:
    """
    Unified ANN index wrapper for HNSW and IVF-PQ.

    Reference: §5 System Architecture, Table 2
    """

    def __init__(self, index_type: str = "hnsw", **kwargs):
        self.index_type = index_type
        self.params = kwargs
        self.vectors = None
        self.index = None

    def build(self, vectors: np.ndarray):
        """Build the ANN index."""
        self.vectors = vectors.astype(np.float32)
        n, d = vectors.shape

        if self.index_type == "hnsw":
            import hnswlib
            self.index = hnswlib.Index(space="l2", dim=d)
            self.index.init_index(
                max_elements=n,
                M=self.params.get("M", 32),
                ef_construction=self.params.get("ef_construction", 200),
            )
            self.index.add_items(self.vectors)
            self.index.set_ef(self.params.get("ef_search", 200))

        elif self.index_type == "ivfpq":
            import faiss
            n_list = self.params.get("n_list", 256)
            n_sub = self.params.get("n_subquantizers", 64)
            n_bits = self.params.get("bits_per_subquantizer", 8)
            quantizer = faiss.IndexFlatL2(d)
            self.index = faiss.IndexIVFPQ(quantizer, d, n_list, n_sub, n_bits)
            self.index.train(self.vectors)
            self.index.add(self.vectors)
            self.index.nprobe = self.params.get("n_probe", 16)

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors.

        Returns (distances, indices).
        """
        query = query.astype(np.float32).reshape(1, -1)

        if self.index_type == "hnsw":
            indices, distances = self.index.knn_query(query, k=k)
            return np.sqrt(distances[0]), indices[0]

        elif self.index_type == "ivfpq":
            distances, indices = self.index.search(query, k)
            return np.sqrt(distances[0]), indices[0]

    def add_vectors(self, new_vectors: np.ndarray):
        """Add vectors to existing index (for decoy insertion)."""
        new_vectors = new_vectors.astype(np.float32)
        self.vectors = np.vstack([self.vectors, new_vectors])

        if self.index_type == "hnsw":
            n_old = self.index.get_current_count()
            self.index.resize_index(n_old + len(new_vectors))
            self.index.add_items(new_vectors, np.arange(n_old, n_old + len(new_vectors)))

        elif self.index_type == "ivfpq":
            self.index.add(new_vectors)
