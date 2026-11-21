import os
import re
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(DATA))

from visualization import visualize_all                       # noqa: E402
from gauss_viz_datasets import DATASETS, main as generate     # noqa: E402

EXE = ROOT / ("run_kmeans_parallel.exe" if os.name == "nt" else "run_kmeans_parallel")


def run_cpp(points_csv, clustered_csv, seed=42):
    result = subprocess.run(
        [str(EXE), str(points_csv), str(clustered_csv), str(seed)],
        capture_output=True, text=True, check=True,
    )
    match = re.search(r"RUNTIME_MS\s+([\d.eE+-]+)", result.stdout)
    return float(match.group(1)) if match else None


def main():
    if not EXE.exists():
        sys.exit(f"{EXE.name} not found. Compile src/cpp/run_kmeans_parallel.cpp first.")

    if not all((DATA / f"{d['name']}_points.csv").exists() for d in DATASETS):
        print("Generating datasets...")
        generate()

    (RESULTS / "clustered").mkdir(parents=True, exist_ok=True)

    runs = []
    for ds in DATASETS:
        name = ds["name"]
        clustered = RESULTS / "clustered" / f"{name}_clustered.csv"
        print(f"Clustering {name} ...")
        runtime = run_cpp(DATA / f"{name}_points.csv", clustered)

        runs.append({
            "name": name, "group": ds["group"], "desc": ds["desc"],
            "n": ds["n"], "k": ds["k"],
            "points": DATA / f"{name}_points.csv",
            "labels": DATA / f"{name}_labels.csv",
            "clustered": clustered,
            "runtime_ms": runtime,
        })

    visualize_all(runs, RESULTS / "figures")


if __name__ == "__main__":
    main()