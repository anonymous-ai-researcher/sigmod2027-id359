"""
Access Control Policy Implementation (§2.1, Definition 2)

Supports RBAC (role-based) and ABAC (attribute-based) paradigms.
"""

import numpy as np


def create_rbac_policy(n_vectors, role_assignments, restricted_roles):
    """
    Create a role-based access control policy.

    Parameters
    ----------
    n_vectors : int
    role_assignments : np.ndarray, shape (n_vectors,)
        Role assigned to each vector.
    restricted_roles : dict
        {user_role: [list of vector roles that are restricted]}
    """
    from .index import AccessControlPolicy
    roles = sorted(restricted_roles.keys())
    policy = AccessControlPolicy(n_vectors, len(roles))

    for i, role in enumerate(roles):
        for restricted_role in restricted_roles[role]:
            mask = role_assignments == restricted_role
            policy.set_restricted(i, np.where(mask)[0])

    return policy
