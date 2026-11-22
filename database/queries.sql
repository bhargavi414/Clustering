-- 1. average metrics for each dataset and algorithm
SELECT
    d.dataset_name,
    e.algorithm,
    AVG(m.runtime_ms) AS avg_runtime,
    AVG(m.inertia) AS avg_inertia,
    AVG(m.ari) AS avg_ari,
    AVG(m.nmi) AS avg_nmi,
    AVG(m.silhouette) AS avg_silhouette,
    AVG(m.davies_bouldin) AS avg_davies_bouldin
FROM datasets d
JOIN experiment_runs e
    ON d.dataset_id = e.dataset_id
JOIN run_metrics m
    ON e.run_id = m.run_id
GROUP BY d.dataset_name, e.algorithm
ORDER BY d.dataset_name, e.algorithm;


-- 2. median silhouette score for each dataset
SELECT
    d.dataset_name,
    e.algorithm,
    PERCENTILE_CONT(0.5)
        WITHIN GROUP (ORDER BY m.silhouette) AS median_silhouette
FROM datasets d
JOIN experiment_runs e
    ON d.dataset_id = e.dataset_id
JOIN run_metrics m
    ON e.run_id = m.run_id
GROUP BY d.dataset_name, e.algorithm;


-- 3. median ARI and interquartile range
SELECT
    d.dataset_name,
    e.algorithm,
    PERCENTILE_CONT(0.5)
        WITHIN GROUP (ORDER BY m.ari) AS median_ari,

    PERCENTILE_CONT(0.75)
        WITHIN GROUP (ORDER BY m.ari)
    -
    PERCENTILE_CONT(0.25)
        WITHIN GROUP (ORDER BY m.ari) AS ari_iqr
FROM datasets d
JOIN experiment_runs e
    ON d.dataset_id = e.dataset_id
JOIN run_metrics m
    ON e.run_id = m.run_id
GROUP BY d.dataset_name, e.algorithm;


-- 4. retrieve all experiment runs for Iris
SELECT
    e.run_id,
    e.algorithm,
    e.seed,
    m.runtime_ms,
    m.ari,
    m.nmi,
    m.silhouette
FROM experiment_runs e
JOIN datasets d
    ON e.dataset_id = d.dataset_id
JOIN run_metrics m
    ON e.run_id = m.run_id
WHERE d.dataset_name = 'iris'
ORDER BY e.seed;


-- 5. examine query execution and index usage
EXPLAIN ANALYZE
SELECT *
FROM experiment_runs
WHERE dataset_id = 1
AND algorithm = 'KMeans||_l=2k_r=5';
