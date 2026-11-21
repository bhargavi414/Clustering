# Scalable and Parallel K-Means

C++ implementations of k-means clustering: sequential, parallel (OpenMP) and with a scalable
initialisation (k-means||), benchmarked against the Scalable K-Means++ paper.

| Part | Files | Reference |
|---|---|---|
| K-Means (random start) and K-Means++ | `kmeans_base.*`, `kmeans_plus_plus.*`, `test_kmeans.cpp` | — |
| **k-means\|\|** (scalable initialisation) + benchmark | `kmeans_parallel.*`, `benchmark_paper.cpp`, `data/gauss_mixture.py`, `src/plot_results.py` | Bahmani et al., *Scalable K-Means++*, PVLDB 5(7), 2012 |
| Multi-threaded Lloyd's k-means (OpenMP) | `kmeans_openMP.*` | — |

k-means has two phases: ① choose starting centers, ② repeat assign + update (Lloyd's iterations).
**k-means\|\|** makes phase ① parallel-friendly; the **OpenMP** versions make phase ② parallel.

---

## Project structure

```
data/
  db.py                      blob datasets (small, medium, large, high_dim, overlap)
  gauss_mixture.py           GaussMixture dataset from the Scalable K-Means++ paper
src/
  preprocessing.py           missing values (column mean) + z-score standardisation
  plot_results.py            graphs for the k-means|| benchmark
  cpp/
    base.hpp                 Point, Cluster, euclidean_distance
    kmeans_base.*            k-means, random start + Lloyd (sequential)
    kmeans_plus_plus.*       k-means++ start
    kmeans_parallel.*        k-means|| start (OpenMP-parallel)
    kmeans_openMP.*          Lloyd's k-means with OpenMP (per-thread point buckets)
    test_kmeans.cpp          K-Means vs K-Means++ (runtime, inertia)
    benchmark_paper.cpp      k-means|| benchmark (quality + scaling)
    main_viz.py
    visualization.py         evaluation and plotting 
results/                     graphs and summary tables
```

---

## 1. Setup

**Windows (sequential builds):**
```bash
py -m pip install numpy scikit-learn matplotlib
```

**WSL / Linux (needed for OpenMP builds):**
```bash
sudo apt install -y build-essential python3-numpy python3-sklearn python3-matplotlib
```
On WSL the project folder is under `/mnt/<drive>/...`; use `python3` instead of `py`.

---

## 2. K-Means vs K-Means++

```bash
py data/db.py                     # choose 1 (predefined) or 2 (custom)
g++ -O2 -std=c++17 -Isrc/cpp src/cpp/test_kmeans.cpp src/cpp/kmeans_base.cpp src/cpp/kmeans_plus_plus.cpp -o test_kmeans.exe
./test_kmeans.exe data/small_points.csv data/medium_points.csv data/large_points.csv data/high_dim_points.csv data/overlap_points.csv
```
Output: `results.csv` (dataset, algorithm, points, dimensions, k, runtime_ms, inertia).
The cluster count is read from the `*_k.txt` file generated next to each dataset.

---

## 3. k-means|| (Scalable K-Means++)

### Algorithm (paper's Algorithm 2) — `src/cpp/kmeans_parallel.cpp`

1. C ← one point chosen uniformly at random; φ = Σ d²(x, C)
2. Repeat r = 5 rounds: keep every point x independently with probability ℓ·d²(x, C)/φ, add the kept points to C, update d² and φ
3. Weight each candidate in C by the number of points nearest to it
4. Weighted k-means++ on C → k starting centers, then Lloyd's iterations

ℓ = 0.5k and ℓ = 2k (the paper's settings). Loops over points use OpenMP
(`parallel for`, `reduction(+:phi)`, one random generator per thread, per-thread lists merged in `critical`).
Compiled without `-fopenmp`, the same code runs sequentially.

`KMeansBase` has a `seed` member (default 42) so repeated runs can use different random seeds.

### Benchmark against the paper's Table 1 (GaussMixture: n = 10,000, d = 15, k = 50, R ∈ {1, 10, 100})

```bash
python3 data/gauss_mixture.py
python3 data/db.py                # option 1 -> large_points.csv for the scaling test
g++ -O2 -std=c++17 -fopenmp -Isrc/cpp src/cpp/benchmark_paper.cpp src/cpp/kmeans_base.cpp src/cpp/kmeans_plus_plus.cpp src/cpp/kmeans_parallel.cpp -o benchmark_paper
./benchmark_paper quality data/gauss_R1_points.csv data/gauss_R10_points.csv data/gauss_R100_points.csv
./benchmark_paper scaling data/large_points.csv
python3 src/plot_results.py
```
- `quality`: 11 runs (seeds 1–11) per algorithm, median cost after initialisation (seed) and after Lloyd (final) → `paper_quality.csv`
- `scaling`: initialisation time vs k, and k-means|| time vs number of threads (median of 5 runs, after one warm-up run) → `paper_scaling.csv`
- On Windows (MinGW without OpenMP), drop `-fopenmp` and use `-std=c++14`; everything runs except the threads test.

### Results (median of 11 runs, cost ÷ 10⁴)

| | R=1 seed | R=1 final | R=10 seed | R=10 final | R=100 seed | R=100 final |
|---|---|---|---|---|---|---|
| Random | 23.0 | 14.5 | 69.8 | 25.5 | 521.3 | 185.1 |
| k-means++ | 23.2 | 14.5 | 49.7 | 23.2 | 112.3 | 52.0 |
| k-means\|\| ℓ=0.5k | 22.1 | 14.5 | 40.0 | 19.7 | 50.2 | 25.5 |
| k-means\|\| ℓ=2k | 22.0 | 14.5 | 40.9 | 20.4 | 62.6 | 31.6 |
| *paper: k-means++* | *23* | *14* | *62* | *31* | *30* | *15* |
| *paper: k-means\|\| ℓ=2k* | *17* | *14* | *27* | *25* | *16* | *15* |

- Same trend as the paper: k-means\|\| gives an equal or lower cost than k-means++, and the gap grows
  as clusters spread out (R = 100: 25.5 vs 52.0).
- Absolute values differ from the paper because the data is randomly generated and Lloyd's iterations stop at 100.
- Parallel speed-up of k-means\|\| initialisation (k = 100, n = 100,000, 16-thread laptop): 226 ms (1 thread) → 38 ms (16 threads), about **6×**.
- k-means\|\| needs r + 2 = 7 passes over the data instead of k passes for k-means++, which is what makes it suitable for distributed data.

Graphs: `results/quality_seed.png`, `quality_final.png`, `init_time_vs_k.png`, `passes_vs_k.png`, `speedup_threads.png`; table: `results/quality_table.csv`.

---

## 4. Preprocessing real datasets

```bash
py src/preprocessing.py raw.csv data/name_points.csv --standardize
```
Replaces missing values with the column mean and (optionally) applies z-score standardisation.
Write the number of clusters into `data/name_points_k.txt` to use the dataset with the benchmarks.

---

### 5. Generate Parallel K-Means Visualization

First, generate the Gaussian mixture visualizations:

```bash
python3 data/gauss_mixture_viz.py
```

Compile the parallel K-Means visualization program.

**Linux / WSL / Windows (MinGW):**

```bash
g++ -O2 -std=c++17 -fopenmp -Isrc/cpp \
src/cpp/run_kmeans_parallel.cpp \
src/cpp/kmeans_base.cpp \
src/cpp/kmeans_parallel.cpp \
-o run_kmeans_parallel
```

**macOS (Homebrew LLVM):**

```bash
/opt/homebrew/opt/llvm/bin/clang++ -O2 -std=c++17 \
-Xpreprocessor -fopenmp \
-I/opt/homebrew/opt/libomp/include \
-L/opt/homebrew/opt/libomp/lib -lomp \
-Isrc/cpp \
src/cpp/run_kmeans_parallel.cpp \
src/cpp/kmeans_base.cpp \
src/cpp/kmeans_parallel.cpp \
-o run_kmeans_parallel
```

Run the visualization script:

```bash
python3 src/main_viz.py
```

On Windows:

```bash
py src/main_viz.py
```

## References

1. B. Bahmani, B. Moseley, A. Vattani, R. Kumar, S. Vassilvitskii. *Scalable K-Means++.* PVLDB 5(7), 2012.
2. L. Dhulipala, J. Lee, J. Łącki, V. Mirrokni. *TeraHAC: Hierarchical Agglomerative Clustering of Trillion-Edge Graphs.* arXiv:2308.03578, 2023 — related idea: few rounds of work so clustering can be distributed.
