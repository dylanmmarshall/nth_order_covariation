**Extracting higher order interactions from generative models of MSAs with multivariate MacLaurin expansion**

This framework extends pairwise sequence saliency analysis to capture 0th order (one below) and 2nd (one above) interactions, expanding on [this work](https://www.biorxiv.org/content/10.1101/2020.11.29.402875v1).

## Overview

Biological sequences exhibit hierarchical coevolutionary structure:

| Order | Tensor | Biology | Math |
|-------|--------|---------|------|
| **0th** | Fields (L, A) | Conservation | log p(x₀) |
| **1st** | Jacobian (L, A, L, A) | Pairwise coevolution (contacts) | ∂ log p / ∂x |
| **2nd** | Hessian (L, A, L, A, L, A) | Triwise epistasis | ∂² log p / ∂x² |

### Key Insight

The MacLaurin expansion of a trained autoencoder f<sub>θ</sub>: ℝ<sup>L×A</sup> → ℝ<sup>L×A</sup> around reference sequence x₀ yields:

```
f(x₀ + h) = f(x₀) + J·h + ½H[h,h] + ⅙T[h,h,h] + ⋯
            └─0th─┘  └1st─┘  └──2nd──┘  └───3rd───┘
```

The **Jacobian J** captures pairwise structure; the symmetrized **Hessian Q** (after S₃ averaging) captures irreducible triwise interactions.

## Architecture

**Models** (all in JAX):
- **MRF** (Markov Random Field / GREMLIN) - Potts model baseline
- **LAE** (Linear Autoencoder) - latent bottleneck
- **VAE** (Variational Autoencoder) - stochastic encoding

**Reduction** (alphabet → position space):
```python
# 0th order: (L, A) → (L) via Frobenius norm
h_i = ||log p[i,:]||_F

# 1st order: (L, A, L, A) → (L, L) via Frobenius norm
J_ij = ||J[i,:,j,:]||_F

# 2nd order: (L, A, L, A, L, A) → (L, L, L) via tensor 2-norm
K_ijk = ||H[i,:,j,:,k,:]||_F
```

## Project Structure

```
HOR/
├── models.py                           # MRF, LAE, VAE + Jacobian/Hessian computation
├── modal_app.py                        # GPU infrastructure (Modal)
├── autoencoders_modal.ipynb            # Main analysis notebook
├── visualization/
│   ├── reduce.py                       # Tensor → position space reduction
│   ├── plot.py                         # Contact map visualization
│   └── tensor_reduction.ipynb          # Interactive Hessian slicing
├── data/
│   ├── AF-P0AA25-F1-msa_v6.npz        # Thioredoxin 1 MSA (19k seqs, L=101)
│   ├── AF-P0AA25-F1-model_v6_contacts.npz  # AlphaFold contacts of Thioredoxin 1
│   └── msa_prep.ipynb                  # Data preprocessing
└── results/                            # Computed Hessians (~7GB each)
```

## Dataset

**AlphaFold Thioredoxin 1**
- Function: Redox protein, reduces disulfide bonds
- Length: 101 amino acids
- MSA: 19,413 sequences (AlphaFold v6)
- Contacts: 462 pairs (< 8Å C-alpha distance)
- Rationale: Bacterial origin (rich coevolutionary signal), monomeric, no cofactors

## Methodology

### S₃ Symmetric Hessian

The raw Hessian H<sub>iajbkc</sub> is asymmetric (directional). To extract undirected triwise interactions, we average over all 6 permutations of the (position, alphabet) pairs:

```python
Q = (H + H¹² + H¹³ + H²³ + H¹²³ + H¹³²) / 6
```

This symmetrization is **model-agnostic** — works for symmetric models (MRF, autoencoders) and asymmetric models (transformers, autoregressive).

## Future Work

**Immediate:**
- Validate on synthetic Potts-sampled MSAs with known h, J, K ground truth
- Develop nth-order APC method (tensor 2-norm + higher-order correction)
- Formalize S<sub>n</sub> symmetrization generalization for arbitrary order (abstract algebra)

**Long-term:**
- Apply to SOTA models (ESM3, MSA Transformer, AlphaFold3)
- Compare extracted triwise interactions to DMS data (triple mutants)
- Quantization + distributed compute for full-length proteins (L > 200)
- Extension to 3rd+ order (computationally challenging)

## Citation

If you use this code, please cite:

```
@software{hor2025,
  author = {Marshall, Dylan},
  title = {Higher-Order Relations: Epistatic Interactions via Multivariate Taylor Expansion},
  year = {2025},
  url = {https://github.com/yourusername/HOR}
}
```

## Related Work

- **[seqsal]** - Pairwise saliency for contact prediction (Jacobian term)
- **Hopf et al. (2017)** - Direct coupling analysis (DCA) for pairwise coevolution
- **Ekeberg et al. (2013)** - GREMLIN (MRF baseline)
- **Koo et al. (2024)** - Attribution analysis for protein models
- **Jumper et al. (2021)** - AlphaFold

## License

MIT

## Contact

For questions or collaboration: dylanmontanamarshall@gmail.com


