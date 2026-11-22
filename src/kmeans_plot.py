from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parent.parent  # script is stored in src/
RESULTS = ROOT / "results"
OUT = RESULTS / "figures"
RUNS = pd.read_csv(RESULTS / "evaluation_runs.csv")

DATASETS = {
    "gaussviz_n3000_R2": "Gaussian R2 (n=3,000)",
    "gaussviz_n15000_R4": "Gaussian R4 (n=15,000)",
    "gaussviz_n50000_R8": "Gaussian R8 (n=50,000)",
    "viz_iris": "Iris (n=150)",
    "viz_moons": "Two Moons (n=3,000)",
}
ORDER = list(DATASETS)

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 220, "font.size": 10,
    "axes.titlesize": 13, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": .22, "grid.linestyle": "--"
})
OUT.mkdir(parents=True, exist_ok=True)

def save(fig, name):
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved:", OUT / name)

def metric_plot(column, title, filename, ylim=None):
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for i, dataset in enumerate(ORDER):
        vals = RUNS.loc[RUNS.dataset == dataset, column].dropna().to_numpy()
        if not len(vals):
            continue
        ax.scatter(i + np.linspace(-.13, .13, len(vals)), vals,
                   s=34, alpha=.75, edgecolor="white", linewidth=.5, zorder=3)
        med = np.median(vals)
        q1, q3 = np.percentile(vals, [25, 75])
        ax.errorbar(i, med, yerr=[[med-q1], [q3-med]], fmt="D",
                    color="black", markersize=6, capsize=5, linewidth=1.7, zorder=5)
    ax.set_xticks(range(len(ORDER)), [DATASETS[d] for d in ORDER],
                  rotation=16, ha="right")
    ax.set_ylabel(column.replace("_", " "))
    ax.set_title(title, loc="left", weight="bold", pad=20)
    fig.text(0.5, 0.965,
             "Dots = individual runs | Diamond and whiskers = median and IQR",
             ha="center", va="top", fontsize=8.5, color="#555")
    if ylim:
        ax.set_ylim(*ylim)
    save(fig, filename)

def runtime_plots():
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for d in ORDER:
        sub = RUNS[RUNS.dataset == d]
        if sub.empty:
            continue
        n = float(sub.points.iloc[0])
        ax.scatter(np.full(len(sub), n), sub.runtime_ms, s=38,
                   alpha=.68, label=DATASETS[d])
        ax.plot([n*.94, n*1.06], [sub.runtime_ms.median()]*2, linewidth=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Number of data points (log scale)")
    ax.set_ylabel("Runtime (ms, log scale)")
    ax.set_title("K-means|| runtime scaling", loc="left", weight="bold")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "05_runtime_scaling.png")

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    data = [RUNS.loc[RUNS.dataset == d, "runtime_ms"].dropna().to_numpy()
            for d in ORDER]
    bp = ax.boxplot(data, patch_artist=True, showfliers=False,
                    medianprops={"color": "black", "linewidth": 2})
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color); patch.set_alpha(.38)
    for i, vals in enumerate(data, 1):
        if len(vals):
            jitter = np.random.default_rng(i).normal(i, .045, len(vals))
            ax.scatter(jitter, vals, s=26, alpha=.65, zorder=3)
    ax.set_xticks(range(1, 6), [DATASETS[d] for d in ORDER],
                  rotation=16, ha="right")
    ax.set_ylabel("Runtime (ms)")
    ax.set_title("Runtime variability across seeds", loc="left", weight="bold")
    save(fig, "06_runtime_variability.png")

def seed_num(path):
    m = re.search(r"seed_(\d+)", path.name)
    return int(m.group(1)) if m else -1

def assignment_files(dataset):
    return sorted((RESULTS / "clustered" / "repeated").glob(f"{dataset}_seed_*_clustered.csv"), key=seed_num)

def load_assignments(dataset):
    result = []
    for file in assignment_files(dataset):
        df = pd.read_csv(file)
        if "cluster_id" in df:
            result.append((seed_num(file), df.cluster_id.to_numpy()))
    return result

def stability_plot(dataset, assignments):
    seeds = [s for s, _ in assignments]
    labels = [x for _, x in assignments]
    if len(labels) < 2:
        return
    mat = np.eye(len(labels))
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            if len(labels[i]) != len(labels[j]):
                raise ValueError(f"Different row counts for {dataset} seed files.")
            mat[i, j] = mat[j, i] = adjusted_rand_score(labels[i], labels[j])
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(seeds)), seeds); ax.set_yticks(range(len(seeds)), seeds)
    ax.set_xlabel("Seed"); ax.set_ylabel("Seed")
    ax.set_title(f"Seed-to-seed assignment stability\n{DATASETS[dataset]}",
                 loc="left", weight="bold")
    for r in range(len(seeds)):
        for c in range(len(seeds)):
            ax.text(c, r, f"{mat[r,c]:.2f}", ha="center", va="center",
                    color="white" if mat[r,c] < .55 else "black", fontsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
    cb.set_label("Adjusted Rand Index")
    save(fig, f"stability_{dataset}.png")

def cluster_size_plot(dataset, assignments):
    if not assignments:
        return
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    max_k = 0
    for seed, labels in assignments:
        pct = pd.Series(labels).value_counts(normalize=True).sort_values(
            ascending=False).to_numpy() * 100
        rank = np.arange(1, len(pct)+1)
        max_k = max(max_k, len(pct))
        ax.plot(rank, pct, marker="o", markersize=3.5,
                linewidth=1.15, alpha=.65, label=f"Seed {seed}")
    ax.set_xlabel("Cluster rank (largest to smallest)")
    ax.set_ylabel("Points in cluster (%)")
    ax.set_xticks(range(1, max_k+1))
    ax.set_title(f"Cluster-size balance across seeds\n{DATASETS[dataset]}",
                 loc="left", weight="bold")
    ax.legend(ncol=2, fontsize=7.5, frameon=False)
    save(fig, f"cluster_sizes_{dataset}.png")

def main():
    required = {"dataset", "points", "runtime_ms", "ARI", "NMI",
                "Silhouette", "Davies_Bouldin"}
    missing = required - set(RUNS.columns)
    if missing:
        raise ValueError(f"evaluation_runs.csv missing columns: {sorted(missing)}")

    metric_plot("ARI", "Adjusted Rand Index (ARI)",
                "01_ari_by_dataset.png", (0, 1.05))
    metric_plot("NMI", "Normalized Mutual Information (NMI)",
                "02_nmi_by_dataset.png", (0, 1.05))
    metric_plot("Silhouette", "Silhouette score",
                "03_silhouette_by_dataset.png")
    metric_plot("Davies_Bouldin", "Davies–Bouldin index (lower is better)",
                "04_davies_bouldin_by_dataset.png")
    runtime_plots()

    pairwise = []
    for d in ORDER:
        assignments = load_assignments(d)
        stability_plot(d, assignments)
        cluster_size_plot(d, assignments)
        for i in range(len(assignments)):
            for j in range(i+1, len(assignments)):
                s1, a = assignments[i]; s2, b = assignments[j]
                if len(a) == len(b):
                    pairwise.append({"dataset": d, "seed_a": s1, "seed_b": s2,
                                     "pairwise_ARI": adjusted_rand_score(a, b)})
    if pairwise:
        pd.DataFrame(pairwise).to_csv(RESULTS / "pairwise_seed_stability.csv",
                                      index=False)
        print("Saved:", RESULTS / "pairwise_seed_stability.csv")
    print("Finished. New PNGs are in results/figures/.")

if __name__ == "__main__":
    main()
