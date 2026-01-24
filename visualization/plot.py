"""Contact map visualization for interaction tensors.

Scatter-style plots similar to protein residue contact maps.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_contact_map_2d(M, ground_truth=None, L_factor=1.0, threshold=None,
                        ax=None, figsize=(6, 6)):
    """Plot (L, L) interaction matrix as contact map scatter plot.

    Style follows models.py plot_contact_map: symmetric scatter with
    ground truth as gray background, predictions overlaid.

    Args:
        M: (L, L) interaction matrix (e.g., reduced Jacobian)
        ground_truth: (L, L) ground truth matrix (optional, shown as gray)
        L_factor: For auto-threshold, use top L*L_factor predictions
        threshold: Explicit threshold (overrides L_factor)
        ax: Matplotlib axis (creates new if None)
        figsize: Figure size if creating new figure

    Returns:
        ax
    """
    M = np.asarray(M)
    L = M.shape[0]

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    # Plot ground truth as gray background
    if ground_truth is not None:
        gt = np.asarray(ground_truth)
        triu_idx = np.triu_indices(L, 1)
        gt_mask = gt[triu_idx] > 0
        ax.scatter(triu_idx[0][gt_mask], triu_idx[1][gt_mask],
                   c='lightgray', s=15, alpha=0.7)
        ax.scatter(triu_idx[1][gt_mask], triu_idx[0][gt_mask],
                   c='lightgray', s=15, alpha=0.7)

    # Determine threshold for predictions
    triu_idx = np.triu_indices(L, 1)
    vals = M[triu_idx]

    if threshold is None:
        top_k = int(L * L_factor)
        threshold = np.sort(vals)[::-1][min(top_k, len(vals) - 1)]

    pred_mask = vals > threshold

    # Plot predictions
    ax.scatter(triu_idx[0][pred_mask], triu_idx[1][pred_mask],
               c='steelblue', s=3, alpha=0.7)
    ax.scatter(triu_idx[1][pred_mask], triu_idx[0][pred_mask],
               c='steelblue', s=3, alpha=0.7)

    ax.set_xlim(0, L)
    ax.set_ylim(L, 0)
    ax.set_aspect('equal')
    ax.set_xlabel('Position')
    ax.set_ylabel('Position')

    return ax
