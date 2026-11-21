"""Datasets for the clustering visualization, all with ground-truth labels:

  group "gaussian": 3 GaussMixture datasets (different R and n)
  group "other":    Iris (standardized) and two moons (non-spherical, hard for k-means)

For each dataset <name> this saves in data/:
    <name>_points.csv        points only (input of the C++ program)
    <name>_points_k.txt      k (read by the C++ program)
    <name>_labels.csv        ground-truth label of every point (used for ARI)
    <name>_ground_truth.csv  points + ground-truth label (the "original dataset with labels")
"""
from pathlib import Path

import numpy as np
from sklearn.datasets import load_iris, make_moons

from gauss_mixture import gauss_mixture

OUT = Path(__file__).resolve().parent

GAUSS_K = 10      # 10 clusters keeps the plots readable (the paper benchmark uses 50)
GAUSS_DIMS = 15   # same dimensionality as the paper benchmark -> PCA is needed to plot


def _gauss(n, R, seed):
    return lambda: gauss_mixture(n=n, k=GAUSS_K, dims=GAUSS_DIMS, R=R, seed=seed,
                                 return_labels=True)


def _iris():
    data = load_iris()
    X = data.data.astype(float)
    X = (X - X.mean(axis=0)) / X.std(axis=0)   # z-score: features have different scales
    return X, data.target


def _moons():
    return make_moons(n_samples=3_000, noise=0.08, random_state=4)


# name -> file prefix, desc -> text used in plot titles
DATASETS = [
    {"name": "gaussviz_n3000_R2",  "group": "gaussian", "desc": "GaussMixture R=2",
     "n": 3_000,  "k": GAUSS_K, "make": _gauss(3_000, 2, 1)},
    {"name": "gaussviz_n15000_R4", "group": "gaussian", "desc": "GaussMixture R=4",
     "n": 15_000, "k": GAUSS_K, "make": _gauss(15_000, 4, 2)},
    {"name": "gaussviz_n50000_R8", "group": "gaussian", "desc": "GaussMixture R=8",
     "n": 50_000, "k": GAUSS_K, "make": _gauss(50_000, 8, 3)},
    {"name": "viz_iris",           "group": "other",    "desc": "Iris (standardized)",
     "n": 150,    "k": 3,       "make": _iris},
    {"name": "viz_moons",          "group": "other",    "desc": "Two moons",
     "n": 3_000,  "k": 2,       "make": _moons},
]


def main():
    for ds in DATASETS:
        X, y = ds["make"]()
        name, dims = ds["name"], X.shape[1]

        header = ",".join([f"dim_{i}" for i in range(dims)] + ["cluster_id"])
        fmt = ["%.6f"] * dims + ["%d"]

        np.savetxt(OUT / f"{name}_points.csv", X, delimiter=",", fmt="%.6f")
        np.savetxt(OUT / f"{name}_labels.csv", y, fmt="%d")
        (OUT / f"{name}_points_k.txt").write_text(str(ds["k"]))
        np.savetxt(OUT / f"{name}_ground_truth.csv", np.column_stack([X, y]),
                   delimiter=",", header=header, comments="", fmt=fmt)

        print(f"saved {name}: n={len(X):,}, d={dims}, k={ds['k']}")


if __name__ == "__main__":
    main()