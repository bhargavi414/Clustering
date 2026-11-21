#pragma once

#include "kmeans_base.hpp"

// k-means|| initialisation (Bahmani et al., VLDB 2012, Algorithm 2)
class KMeansParallel : public KMeansBase {
public:
    double l_factor;  // oversampling factor l = l_factor * k 
    int rounds;       // number of sampling rounds r 
    int num_candidates = 0;  // size of c after sampling

    KMeansParallel(int k_clusters, int max_iterations = 100,
                   double l_factor = 2.0, int rounds = 5)
        : KMeansBase(k_clusters, max_iterations), l_factor(l_factor), rounds(rounds) {}

    void initialise_centers(const vector<Point>& x) override;
};


struct KMeansParallelHalfK : public KMeansParallel {
    KMeansParallelHalfK(int k_clusters, int max_iterations = 100)
        : KMeansParallel(k_clusters, max_iterations, 0.5, 5) {}
};

struct KMeansParallel2K : public KMeansParallel {
    KMeansParallel2K(int k_clusters, int max_iterations = 100)
        : KMeansParallel(k_clusters, max_iterations, 2.0, 5) {}
};
