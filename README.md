**Extracting epistatic interactions from protein sequences via multivariate Taylor expansion**

This framework extends pairwise sequence saliency analysis to capture higher-order epistatic interactions using Jacobian and Hessian tensors of sequence autoencoders.

## Overview

Biological sequences exhibit hierarchical coevolutionary structure:

| Order | Tensor | Biology | Math |
|-------|--------|---------|------|
| **0th** | Fields (L, A) | Sitewise conservation (PSSM) | log p(x₀) |
| **1st** | Jacobian (L, A, L, A) | Pairwise coevolution (contacts) | ∂ log p / ∂x |
| **2nd** | Hessian (L, A, L, A, L, A) | Triwise epistasis | ∂² log p / ∂x² |

Where **L** = sequence length, **A** = alphabet size (20 amino acids).

### Key Insight

The Taylor expansion of a trained autoencoder f<sub>θ</sub>: ℝ<sup>L×A</sup> → ℝ<sup>L×A</sup> around reference sequence x₀ yields:

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
│   ├── AF-P0AA25-F1-model_v6_contacts.npz  # AlphaFold contacts
│   └── msa_prep.ipynb                  # Data preprocessing
└── results/                            # Computed Hessians (~7GB each)
```

## Usage

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Compute Jacobians locally (CPU)
python -c "
from models import train_mrf, mrf_jacobian
import numpy as np

data = np.load('data/AF-P0AA25-F1-msa_v6.npz')
X = data['msa']
params, _ = train_mrf(X, W_seq, n_epochs=200)
J = mrf_jacobian(params, L=101, A=20)
print(f'Jacobian shape: {J.shape}')
"
```

### GPU Compute (Modal)

Hessian computation requires H200/B200 GPU due to memory requirements:

```bash
# Start JupyterLab on H200 GPU
modal run modal_app.py --gpu h200 --timeout-minutes 480

# Upload data to Modal volume (first time only)
modal run modal_app.py --upload

# Open the URL and run autoencoders_modal.ipynb
```

**Memory requirements:**
- Jacobian (L=101): ~100 MB
- Hessian (L=60): ~7 GB (subsectioned MSA)
- Hessian (L=101, full MSA): ~100 GB (requires H200)

## Dataset

**Thioredoxin 1** (E. coli, UniProt: P0AA25)
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

### No APC Correction

For **synthetic Potts-sampled MSAs** (i.i.d. sequences), we use raw Frobenius norm reduction without Average Product Correction (APC). APC is only needed to remove phylogenetic bias in natural sequences.

## Results

### Contact Prediction (1st Order)

| Model | AUC | Notes |
|-------|-----|-------|
| MRF1 | 0.882 | λ=0.01, with bias |
| MRF2 | 0.872 | λ=0.10, no bias |
| LAE1 | 0.810 | rank=512, λ=0.01 |
| LAE2 | 0.315 | needs minibatch training |
| VAE1 | 0.011 | oversmooths pairwise structure |
| VAE2 | 0.119 | deeper architecture |

### Triwise Interactions (2nd Order)

Computed on subsectioned MSA (positions 190-250, L=60):
- 6 Hessian tensors (MRF1, MRF2, LAE1, LAE2, VAE1, VAE2)
- Shape: (60, 20, 60, 20, 60, 20) per model
- Total: ~42 GB stored in `results/20260108/`

Example triwise slice exhibits non-random structure — see `visualization/tensor_reduction.ipynb` for interactive 3D plots.

## Future Work

**Immediate:**
- Validate on synthetic Potts-sampled MSAs with known h, J, K ground truth
- Develop nth-order APC method (tensor 2-norm + higher-order correction)
- Formalize S<sub>n</sub> symmetrization for arbitrary order (abstract algebra)

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

- **seqsal** - Pairwise saliency for contact prediction (Jacobian term)
- **Hopf et al. (2017)** - Direct coupling analysis (DCA) for pairwise coevolution
- **Ekeberg et al. (2013)** - GREMLIN (MRF baseline)
- **Koo et al. (2024)** - Attribution analysis for protein models

## License

MIT

## Contact

For questions or collaboration: dylan@example.com



# CHECKPOINT

results/ generated on 20260123
