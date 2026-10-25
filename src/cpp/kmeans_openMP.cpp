#include "kmeans_openMP.hpp"
#include <random>
#include <limits>
#include<algorithm>
#include <omp.h>

void KMeansopenMP::assign_clusters(vector<Point>& x) {
    for (int i = 0; i < k; i++) {
        clusters[i].points.clear();
    }

    int n = x.size();
    int num_threads = omp_get_max_threads();

    //Thread buckets
    vector<vector<vector<Point>>> local_points(
        num_threads, vector<vector<Point>>(k)
    );

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto& my_points = local_points[tid];

        #pragma omp for schedule(static)
        for (int i = 0; i < n; i++) {
            int curr_cluster = 0;
            double best_dist = euclidean_distance(
                x[i], clusters[0].center
            );

            for (int c = 1; c < k; c++) {
                double d = euclidean_distance(
                    x[i], clusters[c].center
                );

                if (d < best_dist) {
                    best_dist = d;
                    curr_cluster = c;
                }
            }

            x[i].cluster_id = curr_cluster;
            my_points[curr_cluster].push_back(x[i]);
        }
    }

    //Mergin all the threads
    for (int tid = 0; tid < num_threads; tid++) {
        for (int c = 0; c < k; c++) {
            auto& src = local_points[tid][c];
            auto& dest = clusters[c].points;

            dest.insert(dest.end(), src.begin(), src.end());
        }
    }
}

void KMeansopenMP::update_clusters(){
    #pragma omp parallel for schedule(static)
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
    vector<int> pred(x.size());
    int n = x.size();
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        int best_cluster = 0;
        double best_dist = euclidean_distance(x[i], clusters[0].center);

        for (int c = 1; c < k; c++) {
            double d = euclidean_distance(x[i], clusters[c].center);

            if (d < best_dist) {
                best_dist = d;
                best_cluster = c;
            }
        }

        pred[i] = best_cluster;
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
