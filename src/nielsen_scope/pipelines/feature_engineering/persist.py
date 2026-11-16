"""Persist a snapshot of the feature engineering table: a run-stamped 
parquet file on disk, tracked by one row in Postgres so a past snapshot 
can be reloaded to retrain a model without re-running the feature pipeline"""

import logging
from pathlib import Path
from typing import final

import pandas as pd
from sqlalchemy import text, create_engine

from nielsen_scope.db import db_url
from nielsen_scope.pipelines.training.persist import DATA_VERSION, current_run_id

logger = logging.getLogger(__name__)

DATA_VERSION = "nielsen_raw"
SNAPSHOT_DIR = Path("data/04_feature/snapshots")


def write_feature_snapshot(features: pd.DataFrame) -> str:
    """Kedro node: write 'features' to data/04_feature/snapshots/sales_features_<run_id>.parquet
    and record it in feature_snapshots, keyed on the active MLflow run_id"""
    run_id = current_run_id()
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    file_path = SNAPSHOT_DIR / f"sales_features_{run_id}.parquet"
    features.to_parquet(file_path, index=False)

    engine = create_engine(db_url())
    try:
        with engine.begin() as conn:
            conn.execute(
                text("DELETE FROM feature_snapshots WHERE run_id = :run_id"),
                {"run_id": run_id},
            )
            conn.execute(
                text(
                    "INSERT INTO feature_snapshots "
                    "(run_id, file_path, n_rows, n_cols, feature_columns, data_version) "
                    "VALUES (:run_id, :file_path, :n_rows, :n_cols, :feature_columns, :data_version)"
                ),
                {
                    "run_id": run_id,
                    "file_path": str(file_path),
                    "n_rows": len(features),
                    "n_cols": features.shape[1],
                    "feature_columns": ",".join(features.columns),
                    "data_version": DATA_VERSION,
                },
            )
    finally:
        engine.dispose()
    logger.info(
        "write_feature_snapshot: %d rows x %d cols -> %s (run %s)",
        len(features), features.shape[1], file_path, run_id,
    )
    return str(file_path)