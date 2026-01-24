# Nth Order Covariation Determination in MSAs

This repository showcases the determination of the first three orders of covariation within a MSA using a multivariate Maclaurin expansion of a trained autoencoder. This framework extends 1st order pairwise sequence saliency analysis to distinguish 0th order sitewise interactions and 2nd order triwise interactions within a MSA.

```
from models import compute_fields, compute_jacobian, compute_hessian 
# Works with any model: forward_fn(params, x, L, A) -> probs 
h = compute_fields(forward_fn, params, L, A) # (L, A)
J = compute_jacobian(forward_fn, params, L, A) # (L, A, L, A) 
H = compute_hessian(forward_fn, params, L, A)  # (L, A, L, A, L, A)
```

## Citation

If you use this code, please cite:

```
@software{nth_order_covariation2025,
  author = {Marshall, Dylan},
  title = {Extracting Nth Order Interactions Within Multiple Sequence Alignments Using a Multivariate MacLaurin Expansion},
  year = {2025},
  url = {https://github.com/dylanmmarshall/nth_order_covariation}
}
```

## References

- **[Marshall et al.](https://www.biorxiv.org/content/10.1101/2020.11.29.402875v1)** - pairwise saliency
- **[Zhang et al.](https://www.pnas.org/doi/10.1073/pnas.2406285121)** - categorical jacobian, related to pairwise saliency
- **[Varadi et al. (2024)](https://academic.oup.com/nar/article/52/D1/D368/7337620?login=false)** - alphafold database

## License

MIT

Note: not peer-reviewed work














