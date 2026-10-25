# K-Means and K-Means++

C++ implementation comparing K-Means and K-Means++ using runtime and inertia.

### 1. Generate Datasets
```bash
py -m pip install numpy scikit-learn
python data/db.py
```

### 2. Compile
```bash
g++ -g -O0 -std=c++17 -Isrc/cpp src/cpp/test_kmeans.cpp src/cpp/kmeans_base.cpp src/cpp/kmeans_plus_plus.cpp -o test_debug.exe
```

### 3. Run
```bash
./test_debug.exe data/small_points.csv data/medium_points.csv data/large_points.csv data/high_dim_points.csv data/overlap_points.csv
```

### 4. View Results
Open `results.csv` in Excel using Git Bash:

```bash
explorer.exe results.csv
```

Results contain the runtime and inertia for both algorithms.