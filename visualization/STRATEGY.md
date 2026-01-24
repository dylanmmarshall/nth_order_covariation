# Tensor Visualization Strategy

## Overview

Visualize interaction tensors from sequence autoencoders across derivative orders:

| Order | Tensor | Symmetry | Reduction | Output | Interpretation |
|-------|--------|----------|-----------|--------|----------------|
| 0th | (L, A) | S₁ | Frobenius norm over A | (L,) | Fields / biases |
| 1st | (L, A, L, A) | S₂ | Frobenius norm over A dims | (L, L) | Pairwise couplings |
| 2nd | (L, A, L, A, L, A) | S₃ | Tensor 2-norm over A dims | (L, L, L) | Triwise interactions |

## Input Assumptions

Synthetic Potts model data derived from a golden protein dataset after resolving h, J, K (sitewise, pairwise, triwise) relations post average product correction. When sampling using these derived parameters as initial conditions we'll do so in a fashion that assumes the sequences are i.i.d. - therefore no theoretic phylogenetic bias. Therefore the nth-ordder covariate relations derived from this data will not require average product correction post Frobenius norm application.

---

## 0th Order: (L,) Fields

The position-specific scoring matrix (PSSM) or Potts model fields h_i(a). Represents single-site preferences — how much the model favors certain amino acids at each position.

### Source

For autoencoders: `log p(x)` evaluated at x=0, giving (L, A).

### Reduction

```python
def reduce_log_probs(log_p):
    """(L, A) S₁ log-probabilities → (L,) field strengths."""
    return jnp.sqrt(jnp.sum(log_p ** 2, axis=1))
```

### Visualization

Dot plot with position on x-axis, field strength ||h_i|| on y-axis. (TBD)

---

## 1st Order: (L, L) Pairwise Couplings

Reduce S₂-symmetric Jacobian via Frobenius norm over alphabet dimensions, then visualize as a contact map scatter plot (similar to protein residue contact maps).

### Reduction

```python
def reduce_s2_symmetric_jacobian(J):
    """(L, A, L, A) S₂-symmetric Jacobian → (L, L) pairwise couplings."""
    F = jnp.sqrt(jnp.sum(J ** 2, axis=(1, 3)))
    L = F.shape[0]
    return F.at[jnp.diag_indices(L)].set(0.0)
```

### Visualization

Follow the style of `plot_contact_map` in `models.py`:
- Scatter plot with points at interacting position pairs
- Ground truth as gray background, predictions as colored overlay
- Threshold top-L predictions (or use ground truth threshold)
- Symmetric about diagonal

```python
# Reference: models.py plot_contact_map
ax.scatter(triu_idx[0][mask], triu_idx[1][mask], c=color, s=size, alpha=0.7)
ax.scatter(triu_idx[1][mask], triu_idx[0][mask], c=color, s=size, alpha=0.7)
```

---

## 2nd Order: (L, L, L) Triwise Interactions

Reduce S₃-symmetric Hessian via Frobenius norm over alphabet dimensions.

### Reduction

```python
def reduce_s3_symmetric_hessian(H):
    """(L, A, L, A, L, A) S₃-symmetric Hessian → (L, L, L) triwise interactions via the tensor 2-norm."""
    F = jnp.sqrt(jnp.sum(H ** 2, axis=(1, 3, 5)))
    # Zero diagonal planes (i=j, j=k, i=k)
    ...
    return F
```

### Visualization

Analogous 3D scatter plot — position triplets (i, j, k) as points in a cube, with point size/color encoding interaction strength. (TBD)

---

## Implementation

| Function | Input | Output |
|----------|-------|--------|
| `reduce_log_probs(log_p)` | (L,A) S₁ | (L,) |
| `reduce_s2_symmetric_jacobian(J)` | (L,A,L,A) S₂ | (L,L) |
| `reduce_s3_symmetric_hessian(H)` | (L,A,L,A,L,A) S₃ | (L,L,L) |
| `plot_contact_map_2d(M, ...)` | (L,L) | scatter plot |
