#include "kmeans_openMP.hpp"
#include <random>
#include <limits>
#include <omp.h>

void KMeansopenMP::assign_clusters(vector<Point>& x){
    // Clear previously assigned points in clusters
    for (int i = 0; i < k; i++){
        clusters[i].points.clear();
    }

    for (auto& point : x){
        vector<double> dist;
        for (int i = 0; i < k; i++){
            dist.push_back(euclidean_distance(point, clusters[i].center));
        }

        int curr_cluster = min_element(dist.begin(), dist.end()) - dist.begin();
        point.cluster_id = curr_cluster;
        clusters[curr_cluster].points.push_back(point);
    }
}

void KMeansopenMP::update_clusters(){
    for (int i = 0; i < k; i++){
        const auto& pts = clusters[i].points;
        if (!pts.empty()){
            int dims = pts[0].coords.size();
            Point new_center;
            new_center.coords.assign(dims, 0.0);

            for (const auto& p : pts){
                for (int j = 0; j < dims; j++){
                    new_center.coords[j] += p.coords[j];
                }
            }

            for (int j = 0; j < dims; j++){
                new_center.coords[j] /= pts.size();
            }

            clusters[i].center = new_center;
        }
    }
}

vector<int> KMeansopenMP::predict_clusters(const vector<Point>& x){
    vector<int> pred;
    for (const auto& point : x){
        vector<double> dist;
        for (int i = 0; i < k; ++i){
            dist.push_back(euclidean_distance(point, clusters[i].center));
        }
        int best_cluster = min_element(dist.begin(), dist.end()) - dist.begin();
        pred.push_back(best_cluster);
    }
    return pred;
}

void KMeansopenMP::fit(vector<Point>& x){
    initialise_centers(x);

    for (int iter = 0; iter < max_iters; iter++){
        vector<Point> old_centers;
        for (int i = 0; i < k; ++i) {
            old_centers.push_back(clusters[i].center);
        }

        assign_clusters(x);
        update_clusters();

        // Check convergence
        bool converged = true;
        for (int i = 0; i < k; i++){
            if (euclidean_distance(old_centers[i], clusters[i].center) > 1e-4){
                converged = false;
                break;
            }
        }

        if (converged) break;
    }
}

void KMeansopenMP::initialise_centers(const vector<Point>& x){
    if (x.empty()) return;
    int n = x.size();
    mt19937 gen(42);

    uniform_int_distribution<> dis(0, n - 1);
    clusters[0] = Cluster{x[dis(gen)], {}};

    vector<double> min_dists_sq(n, 1e9);

    for (int c = 1; c < k; c++){
        double sum_sq_dist = 0.0;

        //parallelizing distance calculations using openMP
        #pragma omp parallel for reduction(+:sum_sq_dist)
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
