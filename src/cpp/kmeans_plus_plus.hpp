#pragma once

#include "kmeans_base.hpp"

class KMeansPlusPlus : public KMeansBase {
public:
    void initialise_centers(const vector<Point>& X) override;

    KMeansPlusPlus(int k_clusters, int max_iterations = 100)
        : KMeansBase(k_clusters, max_iterations) {}
};