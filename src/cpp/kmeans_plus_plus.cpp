#include "kmeans_plus_plus.hpp"
#include <random>

void KMeansPlusPlus::initialise_centers(const vector<Point>& x){
    if (x.empty()) return;
    int n = x.size();
    mt19937 gen(42);

    uniform_int_distribution<> dis(0, n - 1);
    clusters[0] = Cluster{x[dis(gen)], {}};

    vector<double> min_dists_sq(n, 1e9);

    for (int c = 1; c < k; c++){
        double sum_sq_dist = 0.0;

        for (int i = 0; i < n; i++){
            double d = euclidean_distance(x[i], clusters[c - 1].center);
            min_dists_sq[i] = min(min_dists_sq[i], d*d);
            sum_sq_dist += min_dists_sq[i];
        }

        uniform_real_distribution<> rand_dist(0.0, sum_sq_dist);
        double target = rand_dist(gen);

        double cumulative = 0.0;
        int next_idx = 0;
        for (int i = 0; i < n; i++){
            cumulative += min_dists_sq[i];
            if (cumulative >= target){
                next_idx = i;
                break;
            }
        }

        clusters[c] = Cluster{x[next_idx], {}};
    }
}