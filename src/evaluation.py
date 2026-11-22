import csv
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    davies_bouldin_score
)

#project paths

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

CLUSTER_DIR = RESULTS / "clustered" / "repeated"

RUNS_FILE = RESULTS / "evaluation_runs.csv"
SUMMARY_FILE = RESULTS / "evaluation_summary.csv"

EXE = ROOT / (
    "run_kmeans_parallel.exe"
    if os.name == "nt"
    else "run_kmeans_parallel"
)

SEEDS = list(range(10, 20))

ALGORITHM = "KMeans||_l=2k_r=5"

SILHOUETTE_SAMPLE = 2000

#dataset generation

sys.path.insert(0, str(DATA))
from gauss_viz_datasets import DATASETS, main as generate

#metric calculations

def calculate_inertia(X, labels):

    total = 0.0

    for cluster in np.unique(labels):

        points = X[labels == cluster]
        centroid = np.mean(points, axis=0)

        total += np.sum((points - centroid) ** 2)

    return float(total)


def evaluate(X, actual, predicted):

    if not (len(X) == len(actual) == len(predicted)):
        raise ValueError("Input row counts do not match.")

    if not np.all(np.isfinite(X)):
        raise ValueError("Dataset contains non-finite values.")

    k = len(np.unique(predicted))

    if k < 2 or k >= len(X):
        raise ValueError("Invalid predicted cluster count.")

    #external eval
    ari = adjusted_rand_score(actual, predicted)

    nmi = normalized_mutual_info_score(
        actual, predicted
    )

    #internal eval
    inertia = calculate_inertia(X, predicted)

    silhouette = silhouette_score(
        X,
        predicted,
        sample_size=min(SILHOUETTE_SAMPLE, len(X)),
        random_state=42
    )

    db_index = davies_bouldin_score(X, predicted)

    return {
        "ARI": float(ari),
        "NMI": float(nmi),
        "Inertia": inertia,
        "Silhouette": float(silhouette),
        "Davies_Bouldin": float(db_index)
    }

#cpp clusters

def run_cpp(points_file, output_file, seed):

    result = subprocess.run(
        [
            str(EXE),
            str(points_file),
            str(output_file),
            str(seed)
        ],
        capture_output=True,
        text=True,
        check=True
    )

    match = re.search(
        r"RUNTIME_MS\s+([\d.eE+-]+)",
        result.stdout
    )

    if match:
        return float(match.group(1))

    print("Warning: RUNTIME_MS not found in C++ output.")
    return None


def load_predictions(filename):

    data = np.loadtxt(
        filename,
        delimiter=",",
        skiprows=1,
        ndmin=2
    )

    return data[:, -1].astype(int)


def run_experiments():

    if not EXE.exists():
        raise FileNotFoundError(
            f"{EXE} not found. Compile the C++ program first."
        )

    #generate datasets if required
    if not all(
        (DATA / f"{ds['name']}_points.csv").exists()
        for ds in DATASETS
    ):
        print("Generating datasets...")
        generate()

    CLUSTER_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    all_results = []

    for ds in DATASETS:

        name = ds["name"]

        points_file = DATA / f"{name}_points.csv"
        labels_file = DATA / f"{name}_labels.csv"

        X = np.loadtxt(
            points_file,
            delimiter=",",
            ndmin=2
        )

        actual = np.loadtxt(
            labels_file,
            dtype=int,
            ndmin=1
        )

        print(f"\nDataset: {name}")

        for seed in SEEDS:

            output_file = (
                CLUSTER_DIR /
                f"{name}_seed_{seed}_clustered.csv"
            )

            print(f"Running seed {seed}...")

            runtime = run_cpp(
                points_file,
                output_file,
                seed
            )

            predicted = load_predictions(output_file)

            metrics = evaluate(X, actual, predicted)

            row = {
                "dataset": name,
                "algorithm": ALGORITHM,
                "seed": seed,
                "points": len(X),
                "dimensions": X.shape[1],
                "k": len(np.unique(predicted)),
                "runtime_ms": runtime,
                **metrics
            }

            all_results.append(row)

            print(
                f"ARI={metrics['ARI']:.4f}, "
                f"NMI={metrics['NMI']:.4f}, "
                f"Silhouette={metrics['Silhouette']:.4f}"
            )

    return all_results

def save_runs(results):

    with open(RUNS_FILE, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys()
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"\nIndividual runs saved to {RUNS_FILE}")

#median and iqr

def save_summary(results):

    metrics = [
        "runtime_ms",
        "ARI",
        "NMI",
        "Inertia",
        "Silhouette",
        "Davies_Bouldin"
    ]

    datasets = sorted(set(r["dataset"] for r in results))

    summary = []

    for dataset in datasets:

        rows = [
            r for r in results
            if r["dataset"] == dataset
        ]

        row = {
            "dataset": dataset,
            "algorithm": ALGORITHM,
            "num_runs": len(rows)
        }

        for metric in metrics:

            values = np.array([
                r[metric]
                for r in rows
                if r[metric] is not None
            ], dtype=float)

            if len(values) == 0:
                row[f"{metric}_median"] = None
                row[f"{metric}_IQR"] = None
                continue

            q1, q3 = np.percentile(
                values, [25, 75]
            )

            row[f"{metric}_median"] = float(
                np.median(values)
            )

            row[f"{metric}_IQR"] = float(q3 - q1)

        summary.append(row)

    with open(SUMMARY_FILE, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=summary[0].keys()
        )

        writer.writeheader()
        writer.writerows(summary)

    print(f"Summary saved to {SUMMARY_FILE}")

    print("\nMEDIAN RESULTS")

    for row in summary:

        print(f"\n{row['dataset']}")

        for metric in metrics:

            value = row[f"{metric}_median"]
            iqr = row[f"{metric}_IQR"]

            if value is not None:
                print(
                    f"{metric}: {value:.4f} "
                    f"(IQR: {iqr:.4f})"
                )

def main():

    results = run_experiments()

    save_runs(results)

    save_summary(results)

    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    main()
  
