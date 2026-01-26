"""Sequence models in JAX: MRF, LAE, VAE."""
import jax
import jax.numpy as jnp
from jax import vmap, jit
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import optax


# =============================================================================
# Utilities
# =============================================================================

def get_eff(msa, eff_cutoff=0.8):
    """Compute weight per sequence based on identity clustering."""
    if msa.ndim == 3:
        msa = msa.argmax(-1)
    msa_sm = 1.0 - squareform(pdist(msa, "hamming"))
    return 1.0 / (msa_sm >= eff_cutoff).astype(np.float32).sum(-1)


def con_auc(pred, meas, thresh=0.01):
    """Contact prediction accuracy (top-L AUC)."""
    pred, meas = np.array(pred), np.array(meas)
    eval_idx = np.triu_indices_from(meas, 6)
    pred_, meas_ = pred[eval_idx], meas[eval_idx]
    L_vals = (np.linspace(0.1, 1.0, 10) * len(meas)).astype(int)
    sort_idx = np.argsort(pred_)[::-1]
    return np.mean([(meas_[sort_idx[:l]] > thresh).mean() for l in L_vals])


def pw_contact_map(pw):
    """Convert pairwise saliency to contact map with APC correction."""
    l2 = jnp.sqrt(jnp.sum(pw[:, :20, :, :20] ** 2, axis=(1, 3)))
    L = l2.shape[0]
    l2 = l2.at[jnp.diag_indices(L)].set(0.0)
    row_mean = l2.sum(axis=0)
    apc = jnp.outer(row_mean, row_mean) / row_mean.sum()
    l2_apc = l2 - apc
    return l2_apc.at[jnp.diag_indices(L)].set(0.0)


# =============================================================================
# Plotting
# =============================================================================

def plot_contact_map(contact_maps, L_factor=1, cutoffs=None, sizes=None, colors=None,
                     show_false_positives=True, ax=None):
    """
    Plot contact maps as scatter overlays.

    Args:
        contact_maps: list of (L, L) contact matrices to overlay
            First is typically ground truth, second is prediction
        L_factor: for auto-cutoff, use top L*L_factor predictions
        cutoffs: list of cutoff values (None = auto top-L)
        sizes: list of marker sizes
        colors: list of colors
        show_false_positives: highlight false positives in red (requires 2+ maps)
        ax: matplotlib axis (creates new figure if None)

    Returns:
        ax: matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    n_maps = len(contact_maps)
    if cutoffs is None:
        cutoffs = [0.01] + [None] * (n_maps - 1)
    if sizes is None:
        sizes = [15, 3] + [2] * (n_maps - 2)
    if colors is None:
        colors = ["lightgray", "blue"] + ["green", "orange"][:n_maps - 2]
    max_L = 0
    for i, (con, cutoff, s, c) in enumerate(zip(contact_maps, cutoffs, sizes, colors)):
        con = np.array(con)
        if con.shape[0] > max_L:
            max_L = con.shape[0]
        triu_idx = np.triu_indices_from(con, 1)
        vals = con[triu_idx]
        if cutoff is None:
            top = int(con.shape[0] * L_factor)
            triu_idx_6 = np.triu_indices_from(con, 6)
            vals_6 = con[triu_idx_6]
            cutoff = np.sort(vals_6)[::-1][min(top, len(vals_6) - 1)]
        mask = vals > cutoff
        ax.scatter(triu_idx[0][mask], triu_idx[1][mask], c=c, s=s, alpha=0.7)
        ax.scatter(triu_idx[1][mask], triu_idx[0][mask], c=c, s=s, alpha=0.7)
        if show_false_positives and i == 1 and n_maps >= 2:
            true_con = np.array(contact_maps[0])
            false_pos = (true_con[triu_idx] == 0) & mask
            ax.scatter(triu_idx[0][false_pos], triu_idx[1][false_pos],
                      c="red", s=s, alpha=0.9)
            ax.scatter(triu_idx[1][false_pos], triu_idx[0][false_pos],
                      c="red", s=s, alpha=0.9)
    ax.set_xlim(0, max_L)
    ax.set_ylim(max_L, 0)
    ax.set_aspect("equal")
    ax.set_xlabel("Position")
    ax.set_ylabel("Position")
    return ax


def plot_contact_comparison(true_contacts, predicted_maps, titles=None, figsize=None):
    """
    Plot multiple contact map comparisons in a grid.

    Args:
        true_contacts: (L, L) ground truth contact matrix
        predicted_maps: dict of {name: (L, L) predicted contact matrix}
        titles: optional list of titles
        figsize: figure size

    Returns:
        fig, axes
    """
    n_models = len(predicted_maps)
    if figsize is None:
        figsize = (5 * n_models, 5)
    fig, axes = plt.subplots(1, n_models, figsize=figsize)
    if n_models == 1:
        axes = [axes]
    for ax, (name, pred) in zip(axes, predicted_maps.items()):
        plot_contact_map([true_contacts, pred], ax=ax)
        auc = con_auc(pred, true_contacts)
        ax.set_title(f"{name} (AUC={auc:.3f})")
    plt.tight_layout()
    return fig, axes


def plot_training_curves(losses_dict, figsize=(10, 4)):
    """Plot training loss curves for multiple models."""
    fig, ax = plt.subplots(figsize=figsize)
    for name, losses in losses_dict.items():
        ax.plot(losses, label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.set_title("Training Curves")
    return fig, ax


# =============================================================================
# MRF (Markov Random Field / GREMLIN)
# =============================================================================

def symmetrize_and_mask(W_param, L, A):
    """Make weights symmetric and zero out self-interactions."""
    W_param = (W_param + W_param.T) / 2
    W_4d = W_param.reshape(L, A, L, A)
    mask = 1 - jnp.eye(L)[:, None, :, None]
    return (W_4d * mask).reshape(L * A, L * A)


def mrf_forward(params, x, L, A):
    """MRF forward pass: x -> probs."""
    W_mat = symmetrize_and_mask(params["W"], L, A)
    logits = (W_mat @ x.reshape(-1)).reshape(L, A)
    if "bias" in params:
        logits = logits + params["bias"]
    return jax.nn.softmax(logits, axis=-1)


def mrf_loss(params, X, W_seq, L, A, lam):
    """Pseudolikelihood loss with L2 regularization."""
    def single_loss(x):
        probs = mrf_forward(params, x, L, A)
        return -jnp.sum(x * jnp.log(probs + 1e-8))
    losses = vmap(single_loss)(X)
    weighted_loss = jnp.sum(losses * W_seq) / jnp.sum(W_seq)
    W_c = symmetrize_and_mask(params["W"], L, A)
    reg = lam * (L - 1) * (A - 1) / 2 * jnp.sum(W_c**2) / X.shape[0]
    if "bias" in params:
        reg = reg + lam * jnp.sum(params["bias"] ** 2) / X.shape[0]
    return weighted_loss + reg


def mrf_fields(params, L, A, center=True):
    """Compute 0th-order fields (sitewise conservation) for MRF.

    Returns (L, A) tensor: h[i,a] = log p(x=0)_ia
    This is the constant term in the Maclaurin expansion -- not a derivative.

    Args:
        params: MRF parameters
        L, A: Sequence length and alphabet size
        center: Whether to center over alphabet dimension (gauge fixing)

    Returns:
        (L, A) field tensor representing sitewise preferences
    """
    h = jnp.log(mrf_forward(params, jnp.zeros((L, A)), L, A) + 1e-8)
    if center:
        h = h - h.mean(axis=1, keepdims=True)
    return h


def mrf_jacobian(params, L, A, center=True):
    """Compute symmetrized Jacobian of log-probabilities for MRF.

    Returns (L, A, L, A) tensor: J[i,a,j,b] = d log p(x)_ia / dx_jb
    Symmetrized: (J + J^T) / 2
    """
    def log_prob_fn(x):
        return jnp.log(mrf_forward(params, x, L, A) + 1e-8)
    J = jax.jacfwd(log_prob_fn)(jnp.zeros((L, A)))
    if center:
        for axis in range(4):
            J = J - J.mean(axis=axis, keepdims=True)
    return (J + J.transpose(2, 3, 0, 1)) / 2


def train_mrf(X, W_seq, lam=0.01, use_bias=False, n_epochs=200, verbose=True, seed=42):
    """Train MRF model."""
    N, L, A = X.shape
    lr = 0.1 * np.log(float(W_seq.sum())) / L
    key = jax.random.PRNGKey(seed)
    params = {"W": jax.random.normal(key, (L * A, L * A)) * 0.01}
    if use_bias:
        weighted_freq = jnp.einsum("nla,n->la", X, W_seq) + lam * jnp.log(W_seq.sum())
        init_bias = jnp.log(weighted_freq + 1e-8)
        init_bias = init_bias - init_bias.mean(axis=-1, keepdims=True)
        params["bias"] = init_bias
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)
    @jit
    def step(params, opt_state):
        loss, grads = jax.value_and_grad(
            lambda p: mrf_loss(p, X, W_seq, L, A, lam)
        )(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss
    losses = []
    for epoch in range(n_epochs):
        params, opt_state, loss = step(params, opt_state)
        losses.append(float(loss))
        if verbose and (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch + 1}: loss = {loss:.4f}")
    return params, losses


# =============================================================================
# LAE (Linear Autoencoder)
# =============================================================================

def init_lae_params(key, L, A, rank=256, use_emission=True):
    """Initialize LAE parameters."""
    F = L * A
    keys = jax.random.split(key, 3)
    W_enc = jax.random.normal(keys[0], (F, rank)) * 0.01
    W_dec = jax.random.normal(keys[1], (rank, F)) * 0.01
    params = {"W_enc": W_enc, "W_dec": W_dec}
    if use_emission:
        W_emit = jax.random.normal(keys[2], (A, A)) * 0.01
        params["W_emit"] = W_emit
    return params


def lae_forward(params, x, L, A):
    """LAE forward pass: x -> probs."""
    x_flat = x.reshape(-1)
    z = x_flat @ params["W_enc"]
    logits = z @ params["W_dec"]
    logits = logits.reshape(L, A)
    if "W_emit" in params:
        logits = logits @ params["W_emit"]
    if "bias" in params:
        logits = logits + params["bias"]
    return jax.nn.softmax(logits, axis=-1)


def lae_loss(params, X, W_seq, L, A, lam_w=0.1, lam_e=1.0, lam_b=0.01):
    """LAE loss with L2 regularization."""
    F = L * A
    N = X.shape[0]
    def single_loss(x):
        probs = lae_forward(params, x, L, A)
        return -jnp.sum(x * jnp.log(probs + 1e-8))
    losses = vmap(single_loss)(X)
    weighted_loss = jnp.sum(losses * W_seq) / jnp.sum(W_seq)
    reg = lam_w * F / N * (
        jnp.sum(params["W_enc"]**2) + jnp.sum(params["W_dec"]**2)
    )
    if "W_emit" in params:
        reg = reg + lam_e * jnp.sum(params["W_emit"]**2)
    if "bias" in params:
        reg = reg + lam_b * jnp.sum(params["bias"]**2) / N
    return weighted_loss + reg


def lae_fields(params, L, A, center=True):
    """Compute 0th-order fields (sitewise conservation) for LAE.

    Returns (L, A) tensor: h[i,a] = log p(x=0)_ia
    This is the constant term in the Maclaurin expansion -- not a derivative.

    Args:
        params: LAE parameters
        L, A: Sequence length and alphabet size
        center: Whether to center over alphabet dimension (gauge fixing)

    Returns:
        (L, A) field tensor representing sitewise preferences
    """
    h = jnp.log(lae_forward(params, jnp.zeros((L, A)), L, A) + 1e-8)
    if center:
        h = h - h.mean(axis=1, keepdims=True)
    return h


def lae_jacobian(params, L, A, center=True):
    """Compute symmetrized Jacobian of log-probabilities for LAE.

    Returns (L, A, L, A) tensor: J[i,a,j,b] = d log p(x)_ia / dx_jb
    Symmetrized: (J + J^T) / 2
    """
    def log_prob_fn(x):
        return jnp.log(lae_forward(params, x, L, A) + 1e-8)
    J = jax.jacfwd(log_prob_fn)(jnp.zeros((L, A)))
    if center:
        for axis in range(4):
            J = J - J.mean(axis=axis, keepdims=True)
    return (J + J.transpose(2, 3, 0, 1)) / 2


def train_lae(X, W_seq, rank=256, lam_w=0.1, lam_e=1.0, lam_b=0.01,
              use_emission=True, use_bias=False, n_epochs=135, verbose=True, seed=42):
    """Train LAE model with schedule similar to original."""
    N, L, A = X.shape
    lr = 0.1 * np.log(float(W_seq.sum())) / L
    key = jax.random.PRNGKey(seed)
    params = init_lae_params(key, L, A, rank=rank, use_emission=use_emission)
    if use_bias:
        # Initialize bias from weighted sequence frequencies (like MRF)
        weighted_freq = jnp.einsum("nla,n->la", X, W_seq) + lam_b * jnp.log(W_seq.sum())
        init_bias = jnp.log(weighted_freq + 1e-8)
        init_bias = init_bias - init_bias.mean(axis=-1, keepdims=True)
        params["bias"] = init_bias
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)
    @jit
    def step(params, opt_state):
        loss, grads = jax.value_and_grad(
            lambda p: lae_loss(p, X, W_seq, L, A, lam_w, lam_e, lam_b)
        )(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss
    losses = []
    for epoch in range(n_epochs):
        params, opt_state, loss = step(params, opt_state)
        losses.append(float(loss))
        if verbose and (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch + 1}: loss = {loss:.4f}")
    return params, losses


# =============================================================================
# VAE (Variational Autoencoder)
# =============================================================================

def init_vae_params(key, L, A, enc_dims=[512, 512], rank=32, dec_dims=[512, 512],
                    use_blosum=True):
    """Initialize VAE parameters."""
    F = L * A
    keys = jax.random.split(key, 10)
    key_idx = 0
    params = {}
    dims = [F] + enc_dims
    for i in range(len(enc_dims)):
        params[f"enc_W{i}"] = jax.random.normal(keys[key_idx], (dims[i], dims[i+1])) * 0.01
        params[f"enc_b{i}"] = jnp.zeros(dims[i+1])
        key_idx += 1
    params["mu_W"] = jax.random.normal(keys[key_idx], (enc_dims[-1], rank)) * 0.01
    params["mu_b"] = jnp.zeros(rank)
    key_idx += 1
    params["logvar_W"] = jax.random.normal(keys[key_idx], (enc_dims[-1], rank)) * 0.01
    params["logvar_b"] = jnp.zeros(rank)
    key_idx += 1
    dims = [rank] + dec_dims + [F]
    for i in range(len(dec_dims) + 1):
        params[f"dec_W{i}"] = jax.random.normal(keys[key_idx], (dims[i], dims[i+1])) * 0.01
        params[f"dec_b{i}"] = jnp.zeros(dims[i+1])
        key_idx += 1
    if use_blosum:
        params["blosum_W"] = jax.random.normal(keys[key_idx], (A, A)) * 0.01
        params["blosum_b"] = jnp.zeros(A)
    config = {
        "enc_dims": enc_dims,
        "dec_dims": dec_dims,
        "rank": rank,
        "use_blosum": use_blosum
    }
    return params, config


def vae_encode(params, x, L, A, n_enc_layers):
    """VAE encoder: x -> (mu, logvar)."""
    h = x.reshape(-1)
    for i in range(n_enc_layers):
        h = h @ params[f"enc_W{i}"] + params[f"enc_b{i}"]
        h = jax.nn.selu(h)
    mu = h @ params["mu_W"] + params["mu_b"]
    logvar = h @ params["logvar_W"] + params["logvar_b"]
    return mu, logvar


def vae_decode(params, z, L, A, n_dec_layers, use_blosum):
    """VAE decoder: z -> probs."""
    h = z
    for i in range(n_dec_layers):
        h = h @ params[f"dec_W{i}"] + params[f"dec_b{i}"]
        h = jax.nn.selu(h)
    h = h @ params[f"dec_W{n_dec_layers}"] + params[f"dec_b{n_dec_layers}"]
    h = jax.nn.selu(h)
    logits = h.reshape(L, A)
    if use_blosum:
        logits = logits @ params["blosum_W"] + params["blosum_b"]
    return jax.nn.softmax(logits, axis=-1)


def vae_forward(params, x, L, A, n_enc_layers, n_dec_layers, use_blosum,
                key=None, deterministic=True):
    """VAE forward pass."""
    mu, logvar = vae_encode(params, x, L, A, n_enc_layers)
    if deterministic:
        z = mu
    else:
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(key, mu.shape)
        z = mu + eps * std
    probs = vae_decode(params, z, L, A, n_dec_layers, use_blosum)
    return probs, mu, logvar


def vae_loss(params, X, W_seq, L, A, n_enc_layers, n_dec_layers, use_blosum,
             beta=0.5, key=None):
    """VAE loss: reconstruction + KL divergence."""
    N = X.shape[0]
    if key is None:
        key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, N)
    def single_loss(x, k):
        probs, mu, logvar = vae_forward(params, x, L, A, n_enc_layers, n_dec_layers,
                                         use_blosum, key=k, deterministic=False)
        recon = -jnp.sum(x * jnp.log(probs + 1e-8))
        kl = -0.5 * jnp.sum(1 + logvar - mu**2 - jnp.exp(logvar))
        return recon + beta * kl
    losses = vmap(single_loss)(X, keys)
    weighted_loss = jnp.sum(losses * W_seq) / jnp.sum(W_seq)
    return weighted_loss


def vae_fields(params, L, A, n_enc_layers, n_dec_layers, use_blosum, center=True):
    """Compute 0th-order fields (sitewise conservation) for VAE (deterministic mode).

    Returns (L, A) tensor: h[i,a] = log p(x=0)_ia
    This is the constant term in the Maclaurin expansion -- not a derivative.

    Args:
        params: VAE parameters
        L, A: Sequence length and alphabet size
        n_enc_layers, n_dec_layers: Number of encoder/decoder layers
        use_blosum: Whether BLOSUM emission layer is used
        center: Whether to center over alphabet dimension (gauge fixing)

    Returns:
        (L, A) field tensor representing sitewise preferences
    """
    probs, _, _ = vae_forward(params, jnp.zeros((L, A)), L, A,
                               n_enc_layers, n_dec_layers, use_blosum,
                               deterministic=True)
    h = jnp.log(probs + 1e-8)
    if center:
        h = h - h.mean(axis=1, keepdims=True)
    return h


def vae_jacobian(params, L, A, n_enc_layers, n_dec_layers, use_blosum, center=True):
    """Compute symmetrized Jacobian of log-probabilities for VAE (deterministic mode).

    Returns (L, A, L, A) tensor: J[i,a,j,b] = d log p(x)_ia / dx_jb
    Symmetrized: (J + J^T) / 2
    """
    def log_prob_fn(x):
        probs, _, _ = vae_forward(params, x, L, A, n_enc_layers, n_dec_layers,
                                   use_blosum, deterministic=True)
        return jnp.log(probs + 1e-8)
    J = jax.jacfwd(log_prob_fn)(jnp.zeros((L, A)))
    if center:
        for axis in range(4):
            J = J - J.mean(axis=axis, keepdims=True)
    return (J + J.transpose(2, 3, 0, 1)) / 2


def train_vae(X, W_seq, enc_dims=[512, 512], rank=32, dec_dims=[512, 512],
              use_blosum=True, beta=0.5, n_epochs=400, lr=1e-3,
              verbose=True, seed=42):
    """Train VAE model."""
    N, L, A = X.shape
    n_enc_layers = len(enc_dims)
    n_dec_layers = len(dec_dims)
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    params, config = init_vae_params(init_key, L, A, enc_dims=enc_dims, rank=rank,
                                     dec_dims=dec_dims, use_blosum=use_blosum)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)
    @jit
    def step(params, opt_state, key):
        loss, grads = jax.value_and_grad(
            lambda p: vae_loss(p, X, W_seq, L, A, n_enc_layers, n_dec_layers,
                               use_blosum, beta, key)
        )(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss
    losses = []
    for epoch in range(n_epochs):
        key, subkey = jax.random.split(key)
        params, opt_state, loss = step(params, opt_state, subkey)
        losses.append(float(loss))
        if verbose and (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch + 1}: loss = {loss:.4f}")
    return params, config, losses


# =============================================================================
# Field (Zeroth-Order) Computation
# =============================================================================

def compute_fields(forward_fn, params, L, A, center=True):
    """Compute 0th-order fields (sitewise conservation) for any model.

    Args:
        forward_fn: Function (params, x, L, A) -> probs of shape (L, A)
        params: Model parameters
        L, A: Sequence length and alphabet size
        center: Whether to center over alphabet dimension (gauge fixing)

    Returns:
        (L, A) field tensor: h[i,a] = log p(x=0)_ia
        This is the constant term in the Maclaurin expansion -- not a derivative.
    """
    probs = forward_fn(params, jnp.zeros((L, A)), L, A)
    h = jnp.log(probs + 1e-8)
    if center:
        h = h - h.mean(axis=1, keepdims=True)
    return h

# =============================================================================
# Jacobian (First-Order) Computation
# =============================================================================

def compute_jacobian(forward_fn, params, L, A, center=True):
    """Compute symmetrized Jacobian of log-probabilities for any model.

    Args:
        forward_fn: Function (params, x, L, A) -> probs of shape (L, A)
        params: Model parameters
        L, A: Sequence length and alphabet size
        center: Whether to center along each axis

    Returns:
        (L, A, L, A) tensor: J[i,a,j,b] = d log p(x)_ia / dx_jb
        Symmetrized: (J + J^T) / 2
    """
    def log_prob_fn(x):
        probs = forward_fn(params, x, L, A)
        return jnp.log(probs + 1e-8)
    J = jax.jacfwd(log_prob_fn)(jnp.zeros((L, A)))
    if center:
        for axis in range(4):
            J = J - J.mean(axis=axis, keepdims=True)
    return (J + J.transpose(2, 3, 0, 1)) / 2


# =============================================================================
# Hessian (Second-Order) Computation
# =============================================================================

def _hessian_slice(log_prob_fn, x0, output_idx):
    """Compute Hessian for a single scalar output.

    Args:
        log_prob_fn: Function x -> log_probs of shape (L, A)
        x0: Input point (L, A)
        output_idx: Which flattened output index (0 to L*A-1)

    Returns:
        (L, A, L, A) tensor: H[j,b,k,c] = d^2 log p_i / dx_jb dx_kc
    """
    def scalar_fn(x):
        return log_prob_fn(x).ravel()[output_idx]
    return jax.hessian(scalar_fn)(x0)


def symmetrize_hessian(H):
    """Symmetrize Hessian over all permutations of (position, alphabet) pairs.

    H has shape (L, A, L, A, L, A) with indices (i, a, j, b, k, c).

    Applies S_3 symmetry: averages over all 6 permutations of the three
    (position, alphabet) pairs to extract undirected triwise interactions.

    This is model-agnostic -- works for symmetric models (autoencoders, Potts)
    and asymmetric models (transformers, autoregressive).

    Returns:
        Q[i,a,j,b,k,c] = "strength of irreducible 3-way interaction"
        Symmetric: Q[i,a,j,b,k,c] = Q[j,b,i,a,k,c] = Q[k,c,j,b,i,a] = ...
    """
    H0 = H
    H1 = H.transpose(2, 3, 0, 1, 4, 5)
    H2 = H.transpose(4, 5, 2, 3, 0, 1)
    H3 = H.transpose(0, 1, 4, 5, 2, 3)
    H4 = H.transpose(4, 5, 0, 1, 2, 3)
    H5 = H.transpose(2, 3, 4, 5, 0, 1)
    return (H0 + H1 + H2 + H3 + H4 + H5) / 6.0


def mrf_hessian(params, L, A, center=True, verbose=False):
    """Compute symmetrized Hessian of log-probabilities for MRF.

    Returns (L, A, L, A, L, A) tensor with full S_3 symmetry over (position, alphabet) pairs.
    Q[i,a,j,b,k,c] = "triwise saliency" -- symmetric under all permutations.

    WARNING: Memory scales as O((L*A)^3). Large L may exceed GPU memory.
    """
    def log_prob_fn(x):
        return jnp.log(mrf_forward(params, x, L, A) + 1e-8)
    x0 = jnp.zeros((L, A))
    d = L * A
    slices = []
    for i in range(d):
        if verbose and (i + 1) % 100 == 0:
            print(f"  Hessian slice {i+1}/{d}")
        H_slice = _hessian_slice(log_prob_fn, x0, i)
        slices.append(H_slice.reshape(L, A, L, A))
    H = jnp.stack(slices).reshape(L, A, L, A, L, A)
    if center:
        for axis in range(6):
            H = H - H.mean(axis=axis, keepdims=True)
    return symmetrize_hessian(H)


def lae_hessian(params, L, A, center=True, verbose=False):
    """Compute symmetrized Hessian of log-probabilities for LAE.

    Returns (L, A, L, A, L, A) tensor with full S_3 symmetry over (position, alphabet) pairs.
    Q[i,a,j,b,k,c] = "triwise saliency" -- symmetric under all permutations.

    WARNING: Memory scales as O((L*A)^3). Large L may exceed GPU memory.
    """
    def log_prob_fn(x):
        return jnp.log(lae_forward(params, x, L, A) + 1e-8)
    x0 = jnp.zeros((L, A))
    d = L * A
    slices = []
    for i in range(d):
        if verbose and (i + 1) % 100 == 0:
            print(f"  Hessian slice {i+1}/{d}")
        H_slice = _hessian_slice(log_prob_fn, x0, i)
        slices.append(H_slice.reshape(L, A, L, A))
    H = jnp.stack(slices).reshape(L, A, L, A, L, A)
    if center:
        for axis in range(6):
            H = H - H.mean(axis=axis, keepdims=True)
    return symmetrize_hessian(H)


def vae_hessian(params, L, A, n_enc_layers, n_dec_layers, use_blosum,
                center=True, verbose=False):
    """Compute symmetrized Hessian of log-probabilities for VAE (deterministic mode).

    Returns (L, A, L, A, L, A) tensor with full S_3 symmetry over (position, alphabet) pairs.
    Q[i,a,j,b,k,c] = "triwise saliency" -- symmetric under all permutations.

    WARNING: Memory scales as O((L*A)^3). Large L may exceed GPU memory.
    """
    def log_prob_fn(x):
        probs, _, _ = vae_forward(params, x, L, A, n_enc_layers, n_dec_layers,
                                   use_blosum, deterministic=True)
        return jnp.log(probs + 1e-8)
    x0 = jnp.zeros((L, A))
    d = L * A
    slices = []
    for i in range(d):
        if verbose and (i + 1) % 100 == 0:
            print(f"  Hessian slice {i+1}/{d}")
        H_slice = _hessian_slice(log_prob_fn, x0, i)
        slices.append(H_slice.reshape(L, A, L, A))
    H = jnp.stack(slices).reshape(L, A, L, A, L, A)
    if center:
        for axis in range(6):
            H = H - H.mean(axis=axis, keepdims=True)
    return symmetrize_hessian(H)


def compute_hessian(forward_fn, params, L, A, center=True, verbose=False):
    """Compute symmetrized Hessian of log-probabilities for any model.

    Args:
        forward_fn: Function (params, x, L, A) -> probs of shape (L, A)
        params: Model parameters
        L, A: Sequence length and alphabet size
        center: Whether to center along each axis
        verbose: Print progress

    Returns:
        (L, A, L, A, L, A) tensor with full S_3 symmetry over (position, alphabet) pairs.
        Q[i,a,j,b,k,c] = "triwise saliency" -- symmetric under all permutations.

    WARNING: Memory scales as O((L*A)^3). Large L may exceed GPU memory.
    """
    def log_prob_fn(x):
        probs = forward_fn(params, x, L, A)
        return jnp.log(probs + 1e-8)
    x0 = jnp.zeros((L, A))
    d = L * A
    slices = []
    for i in range(d):
        if verbose and (i + 1) % 100 == 0:
            print(f"  Hessian slice {i+1}/{d}")
        H_slice = _hessian_slice(log_prob_fn, x0, i)
        slices.append(H_slice.reshape(L, A, L, A))
    H = jnp.stack(slices).reshape(L, A, L, A, L, A)
    if center:
        for axis in range(6):
            H = H - H.mean(axis=axis, keepdims=True)
    return symmetrize_hessian(H)
