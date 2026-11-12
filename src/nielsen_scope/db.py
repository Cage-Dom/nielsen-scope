"""The project database URL, read from the environment so it is never a Kedro parameter.

kedro-mlflow logs every `params:*` node input to MLflow, so a URL passed as a
parameter ends up in mlflow.db. Nodes call db_url() at run time instead.
"""

import os

DB_URL_ENV = "NIELSEN_DB_URL"


def db_url() -> str:
    """The SQLAlchemy URL of the project database, from NIELSEN_DB_URL."""
    url = os.environ.get(DB_URL_ENV)
    if not url:
        raise RuntimeError(f"{DB_URL_ENV} is not set: export the project database URL")
    return url