import numpy as np
from sklearn.datasets import make_blobs
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR.mkdir(exist_ok=True)

def generate_dataset(name, n, dimensions, k, spread, seed=42):
    X, _ = make_blobs(
        n_samples=n,
        n_features=dimensions,
        centers=k,
        cluster_std=spread,
        random_state=seed
    )

    filename = OUTPUT_DIR / f"{name}_points.csv"
    np.savetxt(filename, X, delimiter=",", fmt="%.8f")
    (OUTPUT_DIR / f"{name}_points_k.txt").write_text(str(k))

    print(f"\nDataset saved to: {filename}")
    print(f"Points: {n}, Dimensions: {dimensions}, Clusters: {k}")

def predefined_datasets():
    datasets = [
        ("small", 1_000, 2, 5, 1.0),
        ("medium", 10_000, 2, 5, 1.0),
        ("large", 100_000, 2, 5, 1.0),
        ("high_dim", 10_000, 20, 5, 1.0),
        ("overlap", 10_000, 2, 5, 3.0),
    ]

    for dataset in datasets:
        generate_dataset(*dataset)

    print("\nAll predefined datasets generated.")

def interactive_dataset():
    print("\n--- Custom Dataset Generator ---")

    name = input("Dataset name: ").strip()
    if not name or not name.replace("_", "").isalnum():
        print("Invalid name. Use letters, numbers, and underscores.")
        return

    try:
        n = int(input("Number of points: "))
        dimensions = int(input("Number of dimensions: "))
        k = int(input("Number of clusters: "))
        spread = float(input("Cluster spread (e.g. 1.0 or 3.0): "))
        seed = int(input("Random seed (default 42): ") or 42)

        if n < 1 or dimensions < 1 or k < 1 or k > n:
            raise ValueError("Require n >= 1 and 1 <= clusters <= points.")
        if spread <= 0:
            raise ValueError("Cluster spread must be positive.")

        generate_dataset(name, n, dimensions, k, spread, seed)

    except ValueError as e:
        print(f"Invalid input: {e}")

if __name__ == "__main__":
    print("1. Generate predefined datasets")
    print("2. Generate a custom dataset")

    choice = input("Choose an option (1 or 2): ").strip()

    if choice == "1":
        predefined_datasets()
    elif choice == "2":
        interactive_dataset()
    else:
        print("Invalid choice.")
