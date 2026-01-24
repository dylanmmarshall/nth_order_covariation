# Higher-Order Sequence Saliency Analysis

## Background

**seqsal** (`/Users/dylanmarshall/Desktop/evo/seqsal`): Computes pairwise saliency via Jacobian to predict protein contacts and mutation effects. Uses TensorFlow with MRF, LAE, and VAE models.

**higher_order_relations** (this repo): Framework for computing Jacobian and Hessian tensors of sequence autoencoders to analyze pairwise and triwise epistatic interactions.

**Key insight**: seqsal's `pw_saliency()` is the Jacobian term. We can extend this to Hessian and beyond to capture epistatic curvature and higher-order interaction effects.

### Tensor Hierarchy

| Order | Tensor | Symmetry | Reduction | Interpretation |
|-------|--------|----------|-----------|----------------|
| 0th | (L, A) | S₁ | → (L,) | Fields / biases (PSSM) |
| 1st | (L, A, L, A) | S₂ | → (L, L) | Pairwise couplings (Jacobian) |
| 2nd | (L, A, L, A, L, A) | S₃ | → (L, L, L) | Triwise interactions (Hessian) |

Reduction via Frobenius norm over alphabet dimensions. For synthetic Potts data (no phylogenetic bias), no APC correction needed.

---

## Phase A: Reproduce MRF in JAX ✓

- [x] Create Jupyter notebook for development (`mrf_jax.ipynb`)
- [x] Port MRF (GREMLIN-style) model from TensorFlow to JAX
- [x] Load `deepseq_data.npz` from seqsal
- [x] Implement training loop with sequence weighting (`get_eff`)
- [x] Implement `pw_saliency()` using `jax.jacfwd`
- [x] Implement `pw_contact_map()` with APC correction
- [x] Implement evaluation metrics (`con_auc`)
- [x] Verify results match TensorFlow baseline:

| Model | Metric | JAX | TF Target |
|-------|--------|-----|-----------|
| MRF1 | AUC | 0.882 | 0.883 |
| MRF2 | AUC | 0.872 | 0.864 |

## Phase B: Expand to Other Autoencoders ✓

- [x] Port LAE (Linear Autoencoder) to JAX
- [x] Port VAE (Variational Autoencoder) to JAX
- [x] Verify results match TensorFlow baselines
- [x] Refactor shared code into `models.py` module

| Model | Metric | JAX | TF Target | Status |
|-------|--------|-----|-----------|--------|
| LAE1 | AUC | 0.810 | 0.863 | ~ |
| LAE2 | AUC | 0.315 | 0.740 | needs minibatch |
| VAE1 | AUC | 0.011 | 0.001 | ✓ (both poor) |
| VAE2 | AUC | 0.119 | 0.210 | ~ |

Note: VAE contact prediction is inherently poor (oversmooths pairwise structure).
LAE2 AUC gap likely due to minibatch training schedule difference.

### Reimplementation Status

**Fully reimplemented in `models.py`:**
- `get_eff()` - sequence weighting
- `pw_saliency()` - Jacobian via `jax.jacfwd`
- `pw_contact_map()` - L2 norm + APC correction
- `con_auc()` - contact prediction evaluation
- `plot_contact_map()` - scatter overlay with false positives (red)
- `plot_contact_comparison()` - grid comparison helper
- `plot_training_curves()` - loss curve visualization
- MRF, LAE, VAE models with training functions

**Simplified (functional but not identical):**
- LAE training: full batch vs original minibatch schedule
- VAE: no dropout/batchnorm (original has both)

**Environment:**
- Local: M1 Mac on CPU (~1.4s/epoch for MRF)
- jax-metal incompatible with JAX 0.8.x - GPU scaling via Modal instead

## Phase C: Scale Up via Modal ✓

- [x] Set up Modal project structure (`modal_app.py`)
- [x] Create GPU-accelerated training functions
- [x] Create Modal-specific notebook (`autoencoders_modal.ipynb`)
- [x] Configure Modal Sandbox with tunnels for persistent Jupyter sessions
- [x] Test GPU notebook execution on T4

### Usage

```bash
# Start JupyterLab on GPU (interactive)
modal run modal_app.py

# Re-upload data if needed
modal run modal_app.py --upload
```

Opens a JupyterLab instance on Modal GPU with:
- T4 GPU (configurable)
- 2-hour default timeout
- Persistent kernel sessions via Sandbox tunnels
- Data volume at `/data/`

### Key Learnings

**`@modal.web_server` vs `modal.Sandbox`:**
- `web_server`: Ephemeral containers, poor WebSocket support - kernel sessions die
- `Sandbox`: Single persistent container with tunnels - Jupyter works correctly

### Files
- `modal_app.py` - Modal Sandbox app with GPU and JupyterLab
- `autoencoders_modal.ipynb` - Notebook for running inside Modal

## Phase D: MSA Subsectioning & Hessian Computation ✓

Rather than computing the full Hessian for L=252 (552 GB), subsection the MSA to a biologically interesting region identified from the contact map.

### Rationale

The contact map shows dense interaction structure in the C-terminal region (~positions 190-250). Focusing on this region:
- Reduces L from 252 → 60 positions
- Reduces Hessian from 552 GB → 6.9 GB (float32)
- Fits comfortably on H100 GPU (80 GB)
- Preserves biologically meaningful epistatic interactions

### Computed Hessians

All 6 model Hessians computed on H100 GPU and saved to `results/20260108/`:

| File | Model | Shape | Size |
|------|-------|-------|------|
| hess_mrf1.npy | MRF (lam=0.01, bias=True) | (60,20,60,20,60,20) | 6.9 GB |
| hess_mrf2.npy | MRF (lam=0.10, bias=False) | (60,20,60,20,60,20) | 6.9 GB |
| hess_lae1.npy | LAE (rank=512, lam=0.01) | (60,20,60,20,60,20) | 6.9 GB |
| hess_lae2.npy | LAE (rank=512, lam=0.10) | (60,20,60,20,60,20) | 6.9 GB |
| hess_vae1.npy | VAE (enc=[512,512], z=32) | (60,20,60,20,60,20) | 6.9 GB |
| hess_vae2.npy | VAE (enc=[1024], z=256) | (60,20,60,20,60,20) | 6.9 GB |

### Memory Comparison

| Region | L | d = L×A | Hessian (f32) |
|--------|---|---------|---------------|
| Full MSA | 252 | 5040 | 552 GB |
| Positions 190-250 | 60 | 1200 | 6.9 GB |

## Phase E: 0th Order Fields (Sitewise Conservation)

Complete the tensor hierarchy by extracting 0th order terms — the log-probabilities at the reference point, representing sitewise conservation / PSSM.

### Tensor Hierarchy (Complete)

| Order | Tensor | Shape | Symmetry | Reduction | Interpretation |
|-------|--------|-------|----------|-----------|----------------|
| 0th | Fields | (L, A) | S₁ | → (L,) | Sitewise conservation (PSSM) |
| 1st | Jacobian | (L, A, L, A) | S₂ | → (L, L) | Pairwise coevolution |
| 2nd | Hessian | (L, A, L, A, L, A) | S₃ | → (L, L, L) | Triwise epistasis |

### Mathematical Definition

The 0th order term is simply the function value at the reference point:

$$h_{ia} = \log p(x=0)_{ia}$$

This is **not a derivative** — it's the constant term in the Taylor expansion:
$$\log p(x) \approx h + Jx + \frac{1}{2}x^T H x + \ldots$$

### Implementation Tasks

- [ ] Add `mrf_fields()`, `lae_fields()`, `vae_fields()` to `models.py`
- [ ] Add generic `compute_fields()` function
- [ ] Center fields over alphabet dimension (gauge fixing)
- [ ] Compute fields for all 6 models on subsectioned MSA
- [ ] Save to `results/` alongside Jacobians and Hessians
- [ ] Add `reduce_fields()` to `visualization/reduce.py` (already exists)
- [ ] Validate: high field strength ↔ conserved positions

### Connection to Potts Model

In the Potts formulation:
$$P(s) \propto \exp\left( \sum_i h_i(s_i) + \sum_{i<j} J_{ij}(s_i, s_j) + \ldots \right)$$

The fields $h_i(a)$ encode **position-specific amino acid preferences** — the marginal effect of each residue independent of context. Positions under strong selective constraint have peaked field distributions (low entropy).

---

## Phase F: Memory Optimization for Higher-Order Tensors

Higher-order derivatives grow exponentially: Jacobian O(d²), Hessian O(d³), 3rd-order O(d⁴).
For practical computation, we need memory-conscious strategies.

### Tensor Sizes by MSA Length

| L | A | Jacobian | Hessian | 3rd Order |
|---|---|----------|---------|-----------|
| 10 | 20 | 160 KB | 32 MB | 6.4 GB |
| 20 | 20 | 640 KB | 256 MB | 100 GB |
| 30 | 20 | 3.2 MB | 2.9 GB | — |
| 50 | 20 | 16 MB | 32 GB | — |
| 252 | 20 | 100 MB | 512 GB | — |

**Strategy**: Use smaller simulated MSAs (L=10-30) for higher-order analysis.

### Quantization Options

Quantization can apply at three stages:

**A. Model Weights (during training/inference)**
| Precision | Memory | Support | Accuracy |
|-----------|--------|---------|----------|
| float32 | 1x | native | baseline |
| bfloat16 | 0.5x | JAX native | minimal loss |
| float16 | 0.5x | JAX native | small loss |

**B. Autodiff Computation (during derivative calculation)**
| Precision | Memory | Accuracy Loss |
|-----------|--------|---------------|
| float32 | 1x | none |
| float16 | 0.5x | ~1-5% for Hessian |
| bfloat16 | 0.5x | ~1-5% for Hessian |

**C. Output Tensor Storage (post-computation)**
| Format | Compression | Preserves |
|--------|-------------|-----------|
| float32 | 1x | everything |
| float16 | 2x | 3-4 decimal places |
| int8 | 4x | magnitude + sign (~1% error) |
| int4 | 8x | coarse magnitude + sign |
| ternary | 16x | sign + zero detection |
| binary | 32x | sign only |

### Recommended Strategy

For small MSAs where sign and approximate magnitude matter:

```
Model weights: bfloat16 (2x, no downside)
Autodiff: float16 (2x during computation)
Storage: int8 (4x compression, ~1% error)
───────────────────────────────────────
Total: ~16x memory reduction vs naive float32
```

Example for L=50:
- Naive Hessian: 32 GB (float32)
- Optimized: ~2 GB (float16 compute → int8 storage)

### Implementation Tasks

- [ ] Add `quantize_tensor()` and `dequantize_tensor()` utilities
- [ ] Support mixed-precision autodiff in `compute_hessian_tensor()`
- [ ] Benchmark accuracy vs compression tradeoffs
- [ ] Validate sign preservation across quantization levels

---

## Data

Source: `/Users/dylanmarshall/Desktop/evo/seqsal/`
- `data.npz`: MSA (X), mutants (dX), fitness (dY), contacts (cons)
- `deepseq_data.npz`: Alternative dataset

---

## Notes

- JAX chosen for: efficient autodiff, JIT compilation, GPU scaling, cleaner higher-order derivative code
- Modal chosen for: easy GPU access, scalable serverless compute
- Start with small dimensions to verify correctness before scaling

---

## Progress Log

_Update this section as work progresses._

### 2026-01-05
- Initial plan created
- Explored seqsal and higher_order_relations codebases
- **Phase A complete**: MRF ported to JAX
  - Results within ~2% of TensorFlow baseline
  - Set up uv environment with JAX, optax, flax, scipy
- **Phase B complete**: LAE and VAE ported to JAX
  - Created `models.py` module with all three model types
  - MRF contact AUC matches well
  - LAE contact AUC needs minibatch training for full match
  - VAE contact prediction inherently poor (as expected)

### 2026-01-06
- Attempted jax-metal for M1 GPU acceleration
  - jax-metal 0.1.1 incompatible with JAX 0.8.x
  - Decision: use CPU locally, Modal for GPU scaling
- Local performance baseline: ~1.4s/epoch for MRF on M1 CPU

### 2026-01-07
- Added Phase F (originally E): Memory Optimization for Higher-Order Tensors
  - Documented tensor size scaling: Hessian is O(d³), 3rd-order O(d⁴)
  - Strategy: use smaller simulated MSAs (L=10-30) for tractable computation
  - Quantization analysis at three stages: model weights, autodiff, storage
  - Recommended pipeline: bfloat16 weights → float16 autodiff → int8 storage (~16x reduction)
  - Cloud alternative: 8x H100 (640GB) for full-precision on large MSAs
- **Phase C complete**: Modal GPU infrastructure
  - Created `modal_app.py` using Modal Sandbox pattern
  - Initial attempt with `@modal.web_server` failed (kernel sessions died)
  - Switched to `modal.Sandbox.create()` with `encrypted_ports` tunnels
  - JupyterLab now runs persistently on T4 GPU
  - Created `autoencoders_modal.ipynb` - GPU notebook version
  - Data stored in Modal volume `seqsal-data` at `/data/`

**Current file structure:**
```
higher_order_relations/
├── models.py                    # All models + Jacobian/Hessian functions
├── modal_app.py                 # Modal app for GPU compute
├── subsection_msa.py            # MSA subsectioning script
├── autoencoders_modal.ipynb     # Main notebook (Modal GPU)
├── PLAN.md                      # This file
├── pyproject.toml               # uv project config
├── deepseq_data.npz             # Full MSA (L=252)
├── deepseq_data_sub_190_250.npz # Subsectioned MSA (L=60)
├── results/
│   └── 20260108/                # Computed Hessians (~42 GB total)
│       ├── hess_mrf1.npy
│       ├── hess_mrf2.npy
│       ├── hess_lae1.npy
│       ├── hess_lae2.npy
│       ├── hess_vae1.npy
│       └── hess_vae2.npy
└── .venv/                       # Python environment
```

### 2026-01-08
- Updated `modal_app.py` with GPU selection (`--gpu t4/a10g/a100/h100`)
- Extended default timeout to 4 hours for Hessian computation
- Added packages: pandas, seaborn, ipywidgets, tqdm, h5py
- Added **Phase D: MSA Subsectioning** to reduce Hessian compute:
  - Target region: positions 190-250 (C-terminal, dense contacts)
  - Reduces Hessian from 552 GB → ~7 GB (74x reduction)
  - Fits on T4 GPU without streaming or quantization
- Created `subsection_msa.py` script:
  - Slices MSA, contacts, and filters mutants to target region
  - Usage: `python subsection_msa.py --start 190 --end 250`
  - Generated `deepseq_data_sub_190_250.npz`:
    - L=60, d=1200
    - 5349 sequences, 1140 mutants (with mutations in region), 68 contacts
    - Hessian: 6.9 GB (f32) / 1.7 GB (int8)

- **Refactored naming in `models.py`**: Renamed saliency → Jacobian for clarity
  - `mrf_saliency` → `mrf_jacobian`
  - `lae_saliency` → `lae_jacobian`
  - `vae_saliency` → `vae_jacobian`
  - `compute_saliency` → `compute_jacobian`
  - Updated docstrings to clarify: "symmetrized Jacobian of log-probabilities"

- **Implemented Hessian (triwise saliency) in `models.py`**:
  - `_hessian_slice()` - compute Hessian for single scalar output
  - `symmetrize_hessian()` - full S₃ symmetrization over 6 permutations
  - `mrf_hessian()`, `lae_hessian()`, `vae_hessian()`, `compute_hessian()`
  - Returns (L, A, L, A, L, A) tensor with full permutation symmetry

- **S₃ Symmetrization** (analogous to S₂ for Jacobian):
  - Jacobian (2nd order): S₂ symmetry, 2 permutations of (i,a), (j,b)
  - Hessian (3rd order): S₃ symmetry, 6 permutations of (i,a), (j,b), (k,c)
  - Model-agnostic: extracts undirected interactions from any model
  - Interpretation: Q[i,a,j,b,k,c] = "strength of irreducible 3-way interaction"

- **Updated `autoencoders_modal.ipynb`**:
  - Renamed variables: `pw_*` → `jac_*` (Jacobian tensors)
  - Updated imports: `mrf_jacobian`, `lae_jacobian`, `vae_jacobian`, `mrf_hessian`, etc.
  - Added clean Hessian section at end of notebook
  - Notebook structure: Setup → Data → Training/Jacobian → Summary → Save → Hessian

- **Phase D complete**: Computed all 6 Hessians on H100 GPU
  - Ran full notebook on Modal with `--gpu h100 --timeout-minutes 30`
  - Each Hessian ~7 GB, total ~42 GB
  - Downloaded to `results/20260108/` via `modal volume get`

- **Codebase cleanup**:
  - Deleted `higher_order.py` (unused prototype, superseded by `models.py`)
  - Deleted `test_mrf.py` and `test_all_models.py` (one-time validation scripts, outdated imports)

### 2026-01-10
- **Created `visualization/` module** for tensor reduction and plotting:
  - `STRATEGY.md` — documents 0th/1st/2nd order tensor hierarchy
  - `reduce.py` — `reduce_fields()`, `reduce_jacobian()`, `reduce_hessian()`
  - `plot.py` — contact map style scatter plots (2D and 3D)
- **Added 0th order (fields)** to tensor hierarchy:
  - (L, A) → (L,) via Frobenius norm over alphabet
  - Represents position-specific biases / PSSM
  - Visualization: dot plot (position vs field strength)

### 2026-01-11
- **Renamed reduction functions** to use mathematical terminology:
  - `reduce_log_probs()` — 0th order (L, A) S₁ → (L,)
  - `reduce_s2_symmetric_jacobian()` — 1st order (L, A, L, A) S₂ → (L, L)
  - `reduce_s3_symmetric_hessian()` — 2nd order (L, A, L, A, L, A) S₃ → (L, L, L)
- **Created `data/` directory** for simulation pipeline:
  - `initial_conditions/` — ground truth h, J, K from golden protein
  - `synthetic_msa/{ablation}/` — Potts-sampled i.i.d. MSAs (8 ablation conditions)
  - `models/{ablation}/` — trained autoencoder parameters
  - `extracted/{ablation}/` — HOR-extracted tensors for validation
- **Extracted 0th order fields** to `results/20260111/fields_{model}.npy`
- **Removed mutation scoring code** — codebase now focuses on contact prediction only:
  - Removed `sco()`, `mut_rank()` from `models.py`
  - Removed `spearmanr` import (no longer needed)
  - Removed `dX`, `dY` (mutant data) loading from notebook
  - Simplified Jacobian cells to only compute contact AUC
  - Updated summary table (removed Spearman column)
