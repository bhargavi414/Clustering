import csv
import os
from pathlib import Path
import psycopg2

ROOT = Path(__file__).resolve().parent.parent
CSV_FILE = ROOT / "results" / "evaluation_runs.csv"


def connect_db():
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME", "kmeans_project"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.environ["DB_PASSWORD"],
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432")
    )


def insert_results():
    with connect_db() as conn:
        with conn.cursor() as cur:
            with open(CSV_FILE, newline="") as file:
                reader = csv.DictReader(file)

                for row in reader:
                    # Insert dataset
                    cur.execute("""
                        INSERT INTO datasets
                            (dataset_name, num_points, dimensions)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (dataset_name)
                        DO UPDATE SET
                            num_points = EXCLUDED.num_points,
                            dimensions = EXCLUDED.dimensions
                        RETURNING dataset_id;
                    """, (
                        row["dataset"],
                        int(row["num_points"]),
                        int(row["dimensions"])
                    ))

                    dataset_id = cur.fetchone()[0]

                    # Insert experiment run
                    cur.execute("""
                        INSERT INTO experiment_runs
                            (dataset_id, algorithm, seed, k)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (dataset_id, algorithm, seed, k)
                        DO UPDATE SET algorithm = EXCLUDED.algorithm
                        RETURNING run_id;
                    """, (
                        dataset_id,
                        row["algorithm"],
                        int(row["seed"]),
                        int(row["k"])
                    ))

                    run_id = cur.fetchone()[0]

                    #insert eval metrics
                    cur.execute("""
                        INSERT INTO run_metrics
                            (run_id, runtime_ms, inertia, ari, nmi,
                             silhouette, davies_bouldin)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (run_id)
                        DO UPDATE SET
                            runtime_ms = EXCLUDED.runtime_ms,
                            inertia = EXCLUDED.inertia,
                            ari = EXCLUDED.ari,
                            nmi = EXCLUDED.nmi,
                            silhouette = EXCLUDED.silhouette,
                            davies_bouldin = EXCLUDED.davies_bouldin;
                    """, (
                        run_id,
                        float(row["runtime_ms"]) if row["runtime_ms"] else None,
                        float(row["inertia"]),
                        float(row["ari"]),
                        float(row["nmi"]),
                        float(row["silhouette"]),
                        float(row["davies_bouldin"])
                    ))

    print("Results successfully inserted into PostgreSQL!")


if __name__ == "__main__":
    insert_results()
