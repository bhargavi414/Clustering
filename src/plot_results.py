
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("results")
OUT.mkdir(exist_ok=True)

# fixed colour per algorithm (same colour in every graph)
COLORS = {
    "Random": "#2a78d6",
    "KMeans++": "#eb6834",
    "KMeans||_l=0.5k_r=5": "#1baf7a",
    "KMeans||_l=2k_r=5": "#eda100",
}
NICE = {
    "Random": "Random",
    "KMeans++": "k-means++",
    "KMeans||_l=0.5k_r=5": "k-means||  (l = 0.5k)",
    "KMeans||_l=2k_r=5": "k-means||  (l = 2k)",
}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e6e6", "axes.axisbelow": True})


def dataset_label(path):
    name = Path(path).stem.replace("_points", "")
    return name.replace("gauss_R", "R = ")


def plot_quality():
    rows = list(csv.DictReader(open("paper_quality.csv")))
    datasets = list(dict.fromkeys(r["dataset"] for r in rows))
    algos = list(dict.fromkeys(r["algorithm"] for r in rows))
    med = defaultdict(dict)
    for d in datasets:
        for a in algos:
            sel = [r for r in rows if r["dataset"] == d and r["algorithm"] == a]
            med[d][a] = (median(float(r["seed_cost"]) for r in sel) / 1e4,
                         median(float(r["final_cost"]) for r in sel) / 1e4)

    for idx, title in ((0, "Cost right after initialisation (seed)"),
                       (1, "Final cost after Lloyd's iterations")):
        fig, ax = plt.subplots(figsize=(8, 4.6))
        w = 0.8 / len(algos)
        for j, a in enumerate(algos):
            xs = [i + (j - (len(algos) - 1) / 2) * w for i in range(len(datasets))]
            ys = [med[d][a][idx] for d in datasets]
            ax.bar(xs, ys, width=w * 0.92, color=COLORS.get(a), label=NICE.get(a, a))
        ax.set_xticks(range(len(datasets)), [dataset_label(d) for d in datasets])
        ax.set_yscale("log")
        ax.set_ylabel("median cost (÷ 10⁴), log scale  —  lower is better")
        ax.set_title(f"GaussMixture (n=10,000, d=15, k=50): {title}", fontsize=12)
        ax.legend(frameon=False, fontsize=9.5)
        fig.tight_layout()
        fig.savefig(OUT / f"quality_{'seed' if idx == 0 else 'final'}.png", dpi=160)
        plt.close(fig)

    with open(OUT / "quality_table.csv", "w") as f:
        f.write("dataset,algorithm,median_seed_cost_div_1e4,median_final_cost_div_1e4\n")
        for d in datasets:
            for a in algos:
                f.write(f"{dataset_label(d)},{a},{med[d][a][0]:.1f},{med[d][a][1]:.1f}\n")


def plot_scaling():
    rows = list(csv.DictReader(open("paper_scaling.csv")))
    vk = [r for r in rows if r["experiment"] == "vs_k"]
    ks = sorted({int(r["k"]) for r in vk})

    # passes over the data: k-means++ = k passes, k-means|| = 1 + r + 1 = 7 passes (r = 5)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(ks, ks, marker="o", lw=2, color=COLORS["KMeans++"], label="k-means++  (k passes)")
    ax.plot(ks, [7] * len(ks), marker="o", lw=2, color=COLORS["KMeans||_l=2k_r=5"],
            label="k-means||  (r + 2 = 7 passes)")
    ax.set_xlabel("k (number of clusters)")
    ax.set_ylabel("passes over the whole dataset")
    ax.set_title("Why k-means|| scales: passes over the data during initialisation", fontsize=12)
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "passes_vs_k.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.4))
    for a in ("KMeans++", "KMeans||_l=2k_r=5"):
        pts = sorted((int(r["k"]), float(r["init_ms"])) for r in vk if r["algorithm"] == a)
        thr = next(r["threads"] for r in vk if r["algorithm"] == a)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", lw=2, color=COLORS[a],
                label=f"{NICE[a]}  ({thr} thread{'s' if thr != '1' else ''})")
    ax.set_xlabel("k (number of clusters)")
    ax.set_ylabel("initialisation time (ms)")
    ax.set_title("Initialisation time on one machine, in memory (n = 200,000)", fontsize=12)
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "init_time_vs_k.png", dpi=160); plt.close(fig)

    vt = sorted((int(r["threads"]), float(r["init_ms"])) for r in rows if r["experiment"] == "vs_threads")
    if len(vt) > 1:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        base = vt[0][1]
        ax.bar([str(t) for t, _ in vt], [base / ms for _, ms in vt], color=COLORS["KMeans||_l=2k_r=5"], width=0.5)
        for i, (t, ms) in enumerate(vt):
            ax.text(i, base / ms, f"{base / ms:.2f}×\n{ms:.0f} ms", ha="center", va="bottom", fontsize=10)
        ax.set_xlabel("OpenMP threads")
        ax.set_ylabel("speed-up vs 1 thread")
        ax.set_title("k-means|| initialisation: parallel speed-up (k = 100)", fontsize=12)
        ax.set_ylim(0, max(base / ms for _, ms in vt) * 1.3)
        fig.tight_layout(); fig.savefig(OUT / "speedup_threads.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    if Path("paper_quality.csv").exists():
        plot_quality()
    if Path("paper_scaling.csv").exists():
        plot_scaling()
    print("graphs saved in", OUT.resolve())
