"""Database and MLflow run-id helpers shared across pipelines.

kedro-mlflow logs every `params:*` node input to MLflow, so a URL passed as a
parameter ends up in mlflow.db. Nodes call db_url() at run time instead.
"""

import os

import mlflow

DB_URL_ENV = "NIELSEN_DB_URL"


def db_url() -> str:
    """The SQLAlchemy URL of the project database, from NIELSEN_DB_URL."""
    url = os.environ.get(DB_URL_ENV)
    if not url:
        raise RuntimeError(f"{DB_URL_ENV} is not set: export the project database URL")
    return url


def current_run_id() -> str:
    """The id of the active MLflow run (the join key to MLflow)."""
    run = mlflow.active_run()
    if run is None:
        raise RuntimeError("no active MLflow run: run_id is the join key to MLflow")
    return run.info.run_id