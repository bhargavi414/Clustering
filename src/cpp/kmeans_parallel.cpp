#include "kmeans_parallel.hpp"
#include <random>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif


static inline double dist_sq(const Point& a, const Point& b) {
    double s = 0.0;
    for (size_t j = 0; j < a.coords.size(); j++) {
        double diff = a.coords[j] - b.coords[j];
        s += diff * diff;
    }
    return s;
}

// k-means++ where candidate i counts w[i] times
static vector<Point> weighted_kmeans_pp(const vector<Point>& cand, const vector<double>& w,
                                        int k, mt19937& gen) {
    int m = cand.size();
    vector<Point> chosen;

    discrete_distribution<int> first(w.begin(), w.end());
    chosen.push_back(cand[first(gen)]);

    vector<double> d2(m);
    for (int i = 0; i < m; i++) d2[i] = dist_sq(cand[i], chosen[0]);

    while ((int)chosen.size() < k) {
    
        vector<double> score(m);
        double total = 0.0;
        for (int i = 0; i < m; i++) { score[i] = w[i] * d2[i]; total += score[i]; }
        if (total <= 0) break; 

        discrete_distribution<int> pick(score.begin(), score.end());
        const Point& c = cand[pick(gen)];
        chosen.push_back(c);
        for (int i = 0; i < m; i++) d2[i] = min(d2[i], dist_sq(cand[i], c));
    }
    return chosen;
}

void KMeansParallel::initialise_centers(const vector<Point>& x) {
    int n = x.size();
    if (n == 0 || k <= 0) return;
    mt19937 gen(seed);
    double l = l_factor * k;

    // step 1-2: one random center, cost phi
    vector<Point> C;
    C.push_back(x[uniform_int_distribution<int>(0, n - 1)(gen)]);

    vector<double> d2(n);
    double phi = 0.0;  
    #pragma omp parallel for reduction(+:phi)
    for (int i = 0; i < n; i++) { d2[i] = dist_sq(x[i], C[0]); phi += d2[i]; }

   // step 3-6: r rounds, keep each point with prob l*d2/phi
    for (int r = 0; r < rounds && phi > 0; r++) {
        vector<int> picked;

       
        #pragma omp parallel
        {
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            mt19937 tgen(seed * 7919u + r * 104729u + tid); 
            uniform_real_distribution<double> U(0.0, 1.0);
            vector<int> local;

            #pragma omp for schedule(static) nowait
            for (int i = 0; i < n; i++)
                if (U(tgen) < l * d2[i] / phi) local.push_back(i);

            #pragma omp critical
            picked.insert(picked.end(), local.begin(), local.end());
        }
        sort(picked.begin(), picked.end());  
       
        int old_size = C.size();
        for (int idx : picked) C.push_back(x[idx]);

        phi = 0.0;
        #pragma omp parallel for reduction(+:phi) schedule(static)
        for (int i = 0; i < n; i++) {
            for (size_t c = old_size; c < C.size(); c++) d2[i] = min(d2[i], dist_sq(x[i], C[c]));
            phi += d2[i];
        }
    }
    num_candidates = C.size();

    // step 7: weight = number of points nearest to each candidate
    int m = C.size();
    vector<double> w(m, 0.0);
    #pragma omp parallel
    {
        vector<double> local_w(m, 0.0);
        #pragma omp for schedule(static) nowait
        for (int i = 0; i < n; i++) {
            int best = 0;
            double best_d = dist_sq(x[i], C[0]);
            for (int c = 1; c < m; c++) {
                double d = dist_sq(x[i], C[c]);
                if (d < best_d) { best_d = d; best = c; }
            }
            local_w[best] += 1.0;
        }
        #pragma omp critical
        for (int c = 0; c < m; c++) w[c] += local_w[c];
    }

    // step 8: recluster candidates into k centers
    vector<Point> centers = weighted_kmeans_pp(C, w, k, gen);

    uniform_int_distribution<int> any(0, n - 1);
    while ((int)centers.size() < k) centers.push_back(x[any(gen)]);

    clusters.clear();
    for (int i = 0; i < k; i++) {
        Point c = centers[i];
        c.cluster_id = -1;
        clusters[i] = Cluster{c, {}};
    }
}
