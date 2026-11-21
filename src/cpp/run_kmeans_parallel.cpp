// Runs k-means|| (KMeansParallel2K: l = 2k, r = 5) + Lloyd's iterations on one dataset
// and exports the clustered points to CSV.
//
// Usage: run_kmeans_parallel <points.csv> <clustered_output.csv> [seed]
// k is read from <name>_k.txt (same convention as test_kmeans.cpp / benchmark_paper.cpp).

#include "kmeans_base.hpp"
#include "kmeans_parallel.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <chrono>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;
using namespace chrono;

static vector<Point> load_csv(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) throw runtime_error("Cannot open file: " + filename);

    vector<Point> points;
    string line, value;
    while (getline(file, line)) {
        if (line.empty() || line == "\r") continue;
        stringstream ss(line);
        Point p;
        while (getline(ss, value, ',')) p.coords.push_back(stod(value));
        if (!p.coords.empty()) points.push_back(p);
    }
    return points;
}

static int load_k(const string& filename) {
    string kfile = filename;
    kfile.replace(kfile.rfind(".csv"), 4, "_k.txt");
    ifstream f(kfile);
    int k;
    if (!(f >> k)) throw runtime_error("Cannot read k from " + kfile);
    return k;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <points.csv> <clustered_output.csv> [seed]\n";
        return 1;
    }

    try {
        vector<Point> points = load_csv(argv[1]);
        int k = load_k(argv[1]);
        unsigned seed = (argc > 3) ? static_cast<unsigned>(stoul(argv[3])) : 42u;

        if (points.size() < static_cast<size_t>(k)) {
            cerr << "Dataset has fewer points than clusters\n";
            return 1;
        }

        KMeansParallel2K model(k, 100);   // l = 2k, r = 5 rounds
        model.seed = seed;

        auto start = steady_clock::now();
        model.fit(points);                // k-means|| initialisation + Lloyd's iterations
        auto finish = steady_clock::now();
        double runtime_ms = duration<double, milli>(finish - start).count();

        // Row order is preserved: row i of the output == row i of the input.
        export_clusters_to_csv(argv[2], points);

        int threads = 1;
#ifdef _OPENMP
        threads = omp_get_max_threads();
#endif
        // Parsed by main.py
        cout << "THREADS " << threads << "\n";
        cout << "CANDIDATES " << model.num_candidates << "\n";
        cout << "RUNTIME_MS " << runtime_ms << "\n";
    }
    catch (const exception& e) {
        cerr << "Error: " << e.what() << '\n';
        return 1;
    }
    return 0;
}