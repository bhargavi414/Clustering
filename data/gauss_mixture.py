import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent


def gauss_mixture(n=10_000, k=50, dims=15, R=1.0, seed=0, return_labels=False):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0.0, np.sqrt(R), size=(k, dims))   # variance R  -> std sqrt(R)
    labels = rng.integers(0, k, size=n)                       # equal weights for every Gaussian
    X = centers[labels] + rng.normal(0.0, 1.0, size=(n, dims))  # unit variance around each center
    # return_labels=False keeps the old behaviour, so benchmark_paper.cpp's data is unchanged
    return (X, labels) if return_labels else X


if __name__ == "__main__":
    for R in (1, 10, 100):
        X = gauss_mixture(R=R)
        f = OUT / f"gauss_R{R}_points.csv"
        np.savetxt(f, X, delimiter=",", fmt="%.6f")
        (OUT / f"gauss_R{R}_points_k.txt").write_text("50")
        print(f"saved {f.name}: {X.shape[0]} points, {X.shape[1]} dims, k = 50")
