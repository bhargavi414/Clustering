# generates 5 datasets
import numpy as np
from sklearn.datasets import make_blobs
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR.mkdir(exist_ok=True)

datasets = [
    # name, number of points, dimensions, clusters, spread
    ("small",     1_000,   2, 5, 1.0),
    ("medium",   10_000,   2, 5, 1.0),
    ("large",   100_000,   2, 5, 1.0),
    ("high_dim", 10_000,  20, 5, 1.0),
    ("overlap",  10_000,   2, 5, 3.0),
]

for name, n, dimensions, k, spread in datasets:
    X, labels = make_blobs(
        n_samples=n,
        n_features=dimensions,
        centers=k,
        cluster_std=spread,
        random_state=42
    )

    filename = OUTPUT_DIR / f"{name}_points.csv"

    # Save only coordinates: no header and no labels.
    np.savetxt(filename, X, delimiter=",", fmt="%.8f")

    print(
        f"Generated {filename}: "
        f"{n} points, {dimensions} dimensions, {k} clusters"
    )

print("All datasets generated successfully.")
