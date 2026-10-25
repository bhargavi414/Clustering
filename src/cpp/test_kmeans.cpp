
#include "kmeans_base.hpp"
#include "kmeans_plus_plus.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

using namespace std;
using namespace chrono;

vector<Point> load_csv(const string& filename) {
    ifstream file(filename);

    if (!file.is_open()) {
        throw runtime_error("Cannot open file: " + filename);
    }

    vector<Point> points;
    string line;

    while (getline(file, line)) {
        if (line.empty()) continue;

        stringstream ss(line);
        string value;
        Point p;

        while (getline(ss, value, ',')) {
            p.coords.push_back(stod(value));
        }

        if (!p.coords.empty()) {
            points.push_back(p);
        }
    }

    if (!points.empty()) {
        size_t dimensions = points[0].coords.size();

        for (const auto& p : points) {
            if (p.coords.size() != dimensions) {
                throw runtime_error(
                    "Inconsistent dimensions in " + filename
                );
            }
        }
    }

    return points;
}

double calculate_inertia(
    const vector<Point>& points,
    const KMeansBase& model
) {
    double inertia = 0.0;

    for (const auto& p : points) {
        int id = p.cluster_id;

        double d = euclidean_distance(
            p, model.clusters.at(id).center
        );

        inertia += d * d;
    }

    return inertia;
}

template <typename Model>
void test_model(
    const string& filename,
    const vector<Point>& original,
    int k,
    const string& algorithm,
    ofstream& results
) {
    vector<Point> points = original;
    Model model(k, 100);

    auto start = steady_clock::now();
    model.fit(points);
    auto finish = steady_clock::now();

    double runtime_ms =
        duration<double, milli>(finish - start).count();

    double inertia = calculate_inertia(points, model);

    cout << filename
         << " | " << algorithm
         << " | Runtime: " << runtime_ms << " ms"
         << " | Inertia: " << inertia << '\n';

    results << filename << ','
            << algorithm << ','
            << points.size() << ','
            << points[0].coords.size() << ','
            << k << ','
            << runtime_ms << ','
            << inertia << '\n';
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0]
             << " dataset1.csv [dataset2.csv ...]\n";
        return 1;
    }

    ofstream results("results.csv");

    if (!results.is_open()) {
        cerr << "Cannot create results.csv\n";
        return 1;
    }

    results << "dataset,algorithm,points,dimensions,k,"
               "runtime_ms,inertia\n";

    const int k = 5;

    for (int i = 1; i < argc; i++) {
        try {
            vector<Point> points = load_csv(argv[i]);

            if (points.empty() ||
                points.size() < static_cast<size_t>(k)) {
                cerr << "Invalid dataset: " << argv[i] << '\n';
                continue;
            }

            test_model<KMeansBase>(
                argv[i], points, k, "KMeans", results
            );

            test_model<KMeansPlusPlus>(
                argv[i], points, k, "KMeansPlusPlus", results
            );
        }
        catch (const exception& e) {
            cerr << "Error with " << argv[i]
                 << ": " << e.what() << '\n';
        }
    }

    cout << "Results saved to results.csv\n";
    return 0;
}
