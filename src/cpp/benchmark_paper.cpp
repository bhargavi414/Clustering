
#include "kmeans_base.hpp"
#include "kmeans_plus_plus.hpp"
#include "kmeans_parallel.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

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

static double cost(const vector<Point>& x, const KMeansBase& model) {
    double total = 0.0;
    #pragma omp parallel for reduction(+:total)
    for (int i = 0; i < (int)x.size(); i++) {
        double best = 1e300;
        for (const auto& kv : model.clusters) {
            double d = euclidean_distance(x[i], kv.second.center);
            best = min(best, d * d);
        }
        total += best;
    }
    return total;
}

static double median(vector<double> v) {
    sort(v.begin(), v.end());
    return v[v.size() / 2];
}

using Factory = function<unique_ptr<KMeansBase>(int k)>;

static vector<pair<string, Factory>> algorithms() {
    return {
        {"Random",              [](int k) { return make_unique<KMeansBase>(k, 100); }},
        {"KMeans++",            [](int k) { return make_unique<KMeansPlusPlus>(k, 100); }},
        {"KMeans||_l=0.5k_r=5", [](int k) { return make_unique<KMeansParallelHalfK>(k, 100); }},
        {"KMeans||_l=2k_r=5",   [](int k) { return make_unique<KMeansParallel2K>(k, 100); }},
    };
}

static void quality(int argc, char* argv[]) {
    const int RUNS = 11;  
    ofstream out("paper_quality.csv");
    out << "dataset,algorithm,run,k,seed_cost,final_cost,init_ms,total_ms,candidates\n";

    for (int a = 2; a < argc; a++) {
        string file = argv[a];
        vector<Point> data = load_csv(file);
        int k = load_k(file);
        cout << "\n== " << file << "  (n=" << data.size() << ", d=" << data[0].coords.size()
             << ", k=" << k << ") ==\n";

            for (auto& algo : algorithms()) {
            const string& name = algo.first;   
            Factory& make = algo.second;       
            vector<double> seeds, finals;
            for (int run = 1; run <= RUNS; run++) {
                vector<Point> x = data;
                auto model = make(k);
                model->seed = run;

                auto t0 = steady_clock::now();
                model->initialise_centers(x); 
                auto t1 = steady_clock::now();
                double seed_cost = cost(x, *model);

                auto t2 = steady_clock::now();
                model->fit(x);                 
                auto t3 = steady_clock::now();
                double final_cost = cost(x, *model);

                int cand = k;
                if (auto* p = dynamic_cast<KMeansParallel*>(model.get())) cand = p->num_candidates;

                seeds.push_back(seed_cost);
                finals.push_back(final_cost);
                out << file << ',' << name << ',' << run << ',' << k << ',' << seed_cost << ','
                    << final_cost << ',' << duration<double, milli>(t1 - t0).count() << ','
                    << duration<double, milli>(t3 - t2).count() << ',' << cand << '\n';
            }
            printf("  %-22s median seed cost = %12.1f   median final cost = %12.1f\n", name.c_str(),
                   median(seeds), median(finals));
        }
    }
    cout << "\nSaved paper_quality.csv\n";
}

static void scaling(const string& file) {
    vector<Point> data = load_csv(file);
    ofstream out("paper_scaling.csv");
    out << "experiment,algorithm,k,threads,init_ms\n";
    int max_threads = 1;
#ifdef _OPENMP
    max_threads = omp_get_max_threads();
#endif
    auto time_init = [&](KMeansBase& m) {
        vector<double> times;
        for (int rep = 0; rep < 5; rep++) {   
            auto t0 = steady_clock::now();
            m.initialise_centers(data);
            times.push_back(duration<double, milli>(steady_clock::now() - t0).count());
        }
        return median(times);
    };

    
    cout << "Init time vs k on " << file << " (n=" << data.size() << ")\n";
    for (int k : {10, 25, 50, 100, 200}) {
    KMeansParallel2K warmup(10);
    warmup.initialise_centers(data);
        KMeansPlusPlus pp(k);
        KMeansParallel2K par(k);
        double tpp = time_init(pp), tpar = time_init(par);
        out << "vs_k,KMeans++," << k << ",1," << tpp << '\n';
        out << "vs_k,KMeans||_l=2k_r=5," << k << ',' << max_threads << ',' << tpar << '\n';
        printf("  k=%4d   KMeans++ %9.1f ms   KMeans|| %9.1f ms\n", k, tpp, tpar);
    }

#ifdef _OPENMP
    cout << "KMeans|| init time vs threads (k=100)\n";
    for (int t = 1; t <= max_threads; t *= 2) {
        omp_set_num_threads(t);
        KMeansParallel2K par(100);
        double tm = time_init(par);
        out << "vs_threads,KMeans||_l=2k_r=5,100," << t << ',' << tm << '\n';
        printf("  threads=%2d   %9.1f ms\n", t, tm);
    }
#endif
    cout << "Saved paper_scaling.csv\n";
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        cerr << "Usage:\n  " << argv[0] << " quality data1.csv [data2.csv ...]\n  " << argv[0]
             << " scaling data.csv\n";
        return 1;
    }
    string mode = argv[1];
    if (mode == "quality") quality(argc, argv);
    else if (mode == "scaling") scaling(argv[2]);
    else { cerr << "unknown mode " << mode << '\n'; return 1; }
    return 0;
}
