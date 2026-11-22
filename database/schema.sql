-- datasets information
CREATE TABLE IF NOT EXISTS datasets (
    dataset_id SERIAL PRIMARY KEY,
    dataset_name VARCHAR(150) UNIQUE NOT NULL,
    num_points INTEGER NOT NULL CHECK (num_points > 0),
    dimensions INTEGER NOT NULL CHECK (dimensions > 0)
);

-- stores each k-means experiment run
CREATE TABLE IF NOT EXISTS experiment_runs (
    run_id SERIAL PRIMARY KEY,
    dataset_id INTEGER NOT NULL
        REFERENCES datasets(dataset_id),
    algorithm VARCHAR(100) NOT NULL,
    seed INTEGER NOT NULL,
    k INTEGER NOT NULL CHECK (k > 1),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE (dataset_id, algorithm, seed, k)
);

-- stores eval metrics for each run
CREATE TABLE IF NOT EXISTS run_metrics (
    run_id INTEGER PRIMARY KEY
        REFERENCES experiment_runs(run_id)
        ON DELETE CASCADE,

    runtime_ms DOUBLE PRECISION,
    inertia DOUBLE PRECISION NOT NULL,
    ari DOUBLE PRECISION,
    nmi DOUBLE PRECISION,
    silhouette DOUBLE PRECISION,
    davies_bouldin DOUBLE PRECISION
);

-- index to speed up dataset and algorithm searches
CREATE INDEX IF NOT EXISTS idx_runs_dataset_algorithm
ON experiment_runs(dataset_id, algorithm);
