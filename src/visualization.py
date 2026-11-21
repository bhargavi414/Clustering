"""Original (ground truth) vs k-means|| clustered datasets, ARI shown at the top.

Data with more than 2 dimensions is reduced with PCA; 2D data (moons) is plotted as is.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

MAX_PLOT_POINTS = 20_000   # subsample for drawing only; ARI always uses ALL points
CMAP = "tab10"
ALGO = "k-means|| (l = 2k, r = 5)"


# ----------------------------------------------------------------- loading --
def load_points(path):
    return np.loadtxt(path, delimiter=",", ndmin=2)


def load_labels(path):
    return np.loadtxt(path, dtype=int)


def load_predicted_labels(clustered_csv):
    """Clustered CSV = dim_0,...,dim_d-1,cluster_id (header + one row per point)."""
    data = np.loadtxt(clustered_csv, delimiter=",", skiprows=1, ndmin=2)
    return data[:, -1].astype(int)


# --------------------------------------------------------------- utilities --
def align_labels(true, pred):
    """Rename predicted clusters to best match the true ones (Hungarian algorithm).

    Cluster ids are arbitrary; this only makes the same cluster get the same colour in
    both plots. ARI itself is permutation-invariant and uses the raw labels.
    """
    true_ids, pred_ids = np.unique(true), np.unique(pred)
    overlap = np.zeros((len(pred_ids), len(true_ids)))
    for i, p in enumerate(pred_ids):
        for j, t in enumerate(true_ids):
            overlap[i, j] = np.sum((pred == p) & (true == t))

    rows, cols = linear_sum_assignment(-overlap)
    mapping = {pred_ids[r]: true_ids[c] for r, c in zip(rows, cols)}
    next_id = true_ids.max() + 1
    for p in pred_ids:
        if p not in mapping:
            mapping[p] = next_id
            next_id += 1
    return np.array([mapping[p] for p in pred])


def to_2d(X):
    """PCA for d > 2 (returns explained variance); raw coordinates for d == 2."""
    if X.shape[1] == 2:
        return X, None
    pca = PCA(n_components=2, random_state=0)
    return pca.fit_transform(X), pca.explained_variance_ratio_


def _draw_pair(ax_orig, ax_clus, X2, var, true, pred_aligned, run, ari):
    rng = np.random.default_rng(0)
    idx = np.arange(len(X2))
    if len(idx) > MAX_PLOT_POINTS:
        idx = rng.choice(idx, MAX_PLOT_POINTS, replace=False)

    size = 40 if len(X2) <= 500 else (6 if len(X2) <= 3000 else 2)
    kw = dict(s=size, alpha=0.6, cmap=CMAP, vmin=0, vmax=9, linewidths=0)

    ax_orig.scatter(X2[idx, 0], X2[idx, 1], c=true[idx], **kw)
    ax_orig.set_title(f"Original dataset (ground truth)\n"
                      f"{run['desc']}, n={run['n']:,}, k={run['k']}")

    ax_clus.scatter(X2[idx, 0], X2[idx, 1], c=pred_aligned[idx], **kw)
    ax_clus.set_title(f"Clustered by {ALGO}\nARI = {ari:.4f}")

    note = f"ARI = {ari:.4f}"
    if run.get("runtime_ms") is not None:
        note += f"\nRuntime = {run['runtime_ms']:.1f} ms"
    ax_clus.text(0.03, 0.97, note, transform=ax_clus.transAxes, va="top", ha="left",
                 fontsize=10, fontweight="bold",
                 bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9))

    xlim = (X2[:, 0].min(), X2[:, 0].max())
    ylim = (X2[:, 1].min(), X2[:, 1].max())
    for ax in (ax_orig, ax_clus):
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        if var is None:
            ax.set_xlabel("x1")
            ax.set_ylabel("x2")
        else:
            ax.set_xlabel(f"PC1 ({var[0] * 100:.1f}% variance)")
            ax.set_ylabel(f"PC2 ({var[1] * 100:.1f}% variance)")
        ax.grid(alpha=0.2)


# ------------------------------------------------------------------ driver --
def visualize_all(runs, out_dir):
    """runs: list of dicts: name, group, desc, n, k, points, labels, clustered, runtime_ms."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # evaluate + one figure per dataset
    results = []
    for run in runs:
        X = load_points(run["points"])
        true = load_labels(run["labels"])
        pred = load_predicted_labels(run["clustered"])
        if not (len(X) == len(true) == len(pred)):
            raise ValueError(f"Row count mismatch for {run['name']}")

        ari = adjusted_rand_score(true, pred)            # evaluation on all points
        X2, var = to_2d(X)                               # same projection for both plots
        pred_aligned = align_labels(true, pred)

        fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.8))
        _draw_pair(a1, a2, X2, var, true, pred_aligned, run, ari)
        fig.suptitle(f"ARI = {ari:.4f}   |   {ALGO}   |   {run['desc']}, n={run['n']:,}",
                     fontsize=15, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(out_dir / f"{run['name']}_clustering.png", dpi=150)
        plt.close(fig)

        results.append((run, X2, var, true, pred_aligned, ari))
        print(f"{run['name']}: {run['desc']}  n={run['n']:,}  ARI={ari:.4f}")

    # one combined figure per group (gaussian mixtures / iris + moons)
    groups = list(dict.fromkeys(r[0]["group"] for r in results))
    for g in groups:
        items = [r for r in results if r[0]["group"] == g]
        fig, axes = plt.subplots(len(items), 2, figsize=(13, 5.8 * len(items)), squeeze=False)
        for row, (run, X2, var, true, pred_aligned, ari) in enumerate(items):
            _draw_pair(axes[row, 0], axes[row, 1], X2, var, true, pred_aligned, run, ari)
        top = "   |   ".join(f"{it[0]['desc']}: ARI = {it[5]:.4f}" for it in items)
        fig.suptitle(f"{ALGO}\n{top}", fontsize=14, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.97 if len(items) > 2 else 0.93))
        fig.savefig(out_dir / f"all_{g}_clustering.png", dpi=120)
        plt.close(fig)

    # ARI summary bar chart over every dataset
    fig, ax = plt.subplots(figsize=(max(7, 1.8 * len(results)), 4.8))
    labels = [f"{r[0]['desc']}\nn={r[0]['n']:,}" for r in results]
    values = [r[5] for r in results]
    bars = ax.bar(labels, values, color="#1baf7a", width=0.55)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Adjusted Rand Index (higher is better)")
    ax.set_title(f"Clustering quality of {ALGO}")
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "ari_summary.png", dpi=150)
    plt.close(fig)

    print(f"\nFigures saved in: {out_dir}")