"""Tensor reduction functions for visualization.

Reduce higher-order interaction tensors via Frobenius norm over alphabet dimensions.
No APC correction (designed for synthetic Potts model data).

Tensor Hierarchy:
- 0th order: (L, A) log-probabilities → (L,) position field strengths
- 1st order: (L, A, L, A) S₂-symmetric Jacobian → (L, L) pairwise couplings
- 2nd order: (L, A, L, A, L, A) S₃-symmetric Hessian → (L, L, L) triwise interactions
"""

import jax.numpy as jnp


def reduce_log_probs(log_p):
    """Reduce 0th-order log-probabilities to position field strengths.

    Args:
        log_p: (L, A) log-probability tensor (S₁ trivial symmetry)

    Returns:
        (L,) vector of field strengths ||log_p[i,:]||_F per position
    """
    return jnp.sqrt(jnp.sum(log_p ** 2, axis=1))


def reduce_s2_symmetric_jacobian(J):
    """Reduce 1st-order S₂-symmetric Jacobian to pairwise coupling matrix.

    Args:
        J: (L, A, L, A) S₂-symmetric Jacobian tensor

    Returns:
        (L, L) pairwise coupling matrix where (i, j) = ||J[i,:,j,:]||_F
    """
    F = jnp.sqrt(jnp.sum(J ** 2, axis=(1, 3)))
    L = F.shape[0]
    return F.at[jnp.diag_indices(L)].set(0.0)


def reduce_s3_symmetric_hessian(H):
    """Reduce 2nd-order S₃-symmetric Hessian to triwise interaction tensor.

    Args:
        H: (L, A, L, A, L, A) S₃-symmetric Hessian tensor

    Returns:
        (L, L, L) triwise interaction tensor where (i, j, k) = ||H[i,:,j,:,k,:]||_F
    """
    F = jnp.sqrt(jnp.sum(H ** 2, axis=(1, 3, 5)))
    L = F.shape[0]

    # Zero diagonal planes (i=j, j=k, i=k)
    for i in range(L):
        F = F.at[i, i, :].set(0.0)
        F = F.at[i, :, i].set(0.0)
        F = F.at[:, i, i].set(0.0)

    return F
