#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <iostream>

using namespace std;

struct Point{
    vector<double> coords;
    int cluster_id = -1;
};

struct Cluster{
    Point center;
    vector<Point> points;
};

inline double euclidean_distance(const Point& a, const Point& b){
    double dist = 0.0;
    for(int i = 0; i < a.coords.size(); i++){
        dist += (a.coords[i] - b.coords[i]) * (a.coords[i] - b.coords[i]);
    }

    return sqrt(dist);
}

inline void export_clusters_to_csv (const string& filename, const vector<Point>& points){
    ofstream file(filename);

    if(file.is_open() == false){
        cerr << "Error in opening file " << filename << endl;
        return;
    }
    
    int dims = points[0].coords.size();
    for(int i = 0; i < dims; i++){
        file << "dim_" << i << ",";
    }
    file << "cluster_id\n";

    for(const auto& point : points){
        for(int i = 0; i < point.coords.size(); i++){
            file << point.coords[i] << ",";
        }
        file << point.cluster_id << "\n";
    }

    file.close();
}