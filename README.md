# Nth Order Covariation Determination in MSAs

This repository is an attempt at determining the first three orders of covariation within a multiple sequence alignment (MSA) using a multivariate Maclaurin expansion of a trained autoencoder. A suite of autoencoders are evaluated. This framework extends 1st order pairwise sequence saliency analysis to distinguish 0th order sitewise interactions and 2nd order triwise interactions within a MSA.

```Python
from models import compute_fields, compute_jacobian, compute_hessian
# Works with any model: forward_fn(params, x, L, A) -> probs
h = compute_fields(forward_fn, params, L, A) # (L, A)
J = compute_jacobian(forward_fn, params, L, A) # (L, A, L, A)
H = compute_hessian(forward_fn, params, L, A)  # (L, A, L, A, L, A)
```

```
nth_order_covariation/
├── models.py                       # MRF, LAE, VAE + derivative extraction
├── modal_app.py                    # GPU training orchestration
├── autoencoders_modal.ipynb        # Training & tensor extraction
├── data/                           # MSA, contacts, structure
├── results/                        # Fields, Jacobians, Hessians
├── visualization/
│   ├── modal_app.py                # GPU analysis orchestration
│   └── tensor_reduction_full.ipynb # Conclusion
└── LICENSE                         # Apache 2.0
```

Ongoing: math and code for 3rd order + relations

## References

- **[Marshall et al. (2020)](https://www.biorxiv.org/content/10.1101/2020.11.29.402875v1)** - pairwise saliency
- **[Zhang et al. (2024)](https://www.pnas.org/doi/10.1073/pnas.2406285121)** - categorical jacobian, related to pairwise saliency
- **[Varadi et al. (2024)](https://academic.oup.com/nar/article/52/D1/D368/7337620?login=false)** - alphafold database

---

## License

[Apache License 2.0](LICENSE).

---

Note: not peer-reviewed work

