#include <kmeans_base.hpp>
#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

void KMeansBase::initialise_centers(const vector<Point>& x) {
    if (x.empty() or k <= 0) return;

    mt19937 gen(10);
    vector<int> indices(x.size());

    iota(indices.begin(), indices.end(), 0);
    shuffle(indices.begin(), indices.end(), gen);

    for (int i = 0; i < k; i++) {
        clusters[i] = Cluster{x[indices[i]], {}};
    }
}

void KMeansBase::assign_clusters(vector<Point>& x){
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

void KMeansBase::update_clusters(){
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

vector<int> KMeansBase::predict_clusters(const vector<Point>& x){
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

void KMeansBase::fit(vector<Point>& x){
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