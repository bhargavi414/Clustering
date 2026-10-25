#pragma once

#include "base.hpp"
#include <map>

class KMeansopenMP{
public:
    int k;
    int max_iters;
    map<int, Cluster> clusters;

    //it is a virtual function so that other child class (Kmeans++) can override this function
    virtual void initialise_centers(const vector<Point>& x);

    KMeansopenMP(int k_clusters, int max_iteration = 100) : k(k_clusters), max_iters(max_iteration) {}

    virtual ~KMeansopenMP() = default;

    void assign_clusters(vector<Point>& x);
    void update_clusters();
    vector<int> predict_clusters(const vector<Point>& x);

    void fit(vector<Point>& x);

    const map<int, Cluster>& get_clusters() const {return clusters;}
};