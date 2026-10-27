"""Persist walk-forward results to Postgres: the forecasts and model_metrics tables."""

import logging

import mlflow
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection

logger = logging.getLogger(__name__)

DATA_VERSION = "nielsen_raw"
KIND_BACKTEST = "backtest"
TABLE_KEYS = {
    "forecasts": ["isbn", "model_type", "run_id"],
    "model_metrics": ["isbn", "model_type", "run_id"],
}
# cross_validate metrics dict key -> model_metrics.metric (fold NULL)
AGGREGATE_METRICS = {
    "mae_mean": "mae",
    "mae_std": "mae_std",
    "rmse_mean": "rmse",
    "rmse_std": "rmse_std",
    "mape_mean": "mape",
    "mape_std": "mape_std",
    "relative_mae": "rel_mae",
    "mape_excluded_rows": "mape_excluded_rows",
}
# cross_validate fold_metrics column -> model_metrics.metric (one row per fold)
FOLD_METRICS = ["mae", "rmse", "mape", "rel_mae", "mape_excluded_rows"]


def current_run_id() -> str:
    """The id of the active MLflow run (the join key to MLflow)."""
    run = mlflow.active_run()
    if run is None:
        raise RuntimeError("no active MLflow run: run_id is the join key to MLflow")
    return run.info.run_id


def build_forecast_rows(predictions: pd.DataFrame, params: dict, run_id: str) -> pd.DataFrame:
    """Backtest rows for the forecasts table, from cross_validate's predictions.

    Negative y_true is accepted on purpose (A-070): volume is modelled as recorded.
    """
    cv = params["cv"]
    expected = cv["n_splits"] * cv["test_horizon"]
    if len(predictions) != expected:
        raise ValueError(f"expected {expected} backtest rows, got {len(predictions)}")
    if predictions["end_date"].isna().any():
        raise ValueError("backtest rows need end_date: pass dates to cross_validate")
    if not predictions["end_date"].is_unique:
        raise ValueError("backtest end_date values are not unique")
    if not np.isfinite(predictions["y_pred"]).all():
        raise ValueError("y_pred has NaN or infinite values")
    if (predictions["y_true"] % 1 != 0).any():
        raise ValueError("y_true has non-integer values")

    return pd.DataFrame(
        {
            "isbn": str(params["isbn"]),
            "model_type": params["model_type"],
            "run_id": run_id,
            "kind": KIND_BACKTEST,
            "fold": predictions["fold"].astype("int16"),
            "end_date": pd.to_datetime(predictions["end_date"]).dt.date,
            "y_true": predictions["y_true"].astype("int64"),
            "y_pred": predictions["y_pred"].astype(float),
            "lo_95": predictions["lo_95"].astype(float),
            "hi_95": predictions["hi_95"].astype(float),
            "data_version": DATA_VERSION,
        }
    )


def build_metric_rows(
    metrics: dict, fold_metrics: pd.DataFrame, params: dict, run_id: str
) -> pd.DataFrame:
    """Long-format rows for the model_metrics table: one per fold and metric, plus
    the aggregates (fold NULL). fell_back is the fold's flag on fold rows and
    'any fold fell back' on aggregate rows; NULL when the model cannot fall back."""
    keys = {
        "isbn": str(params["isbn"]),
        "model_type": params["model_type"],
        "run_id": run_id,
        "data_version": DATA_VERSION,
    }
    rows = []
    for fold_row in fold_metrics.itertuples(index=False):
        for metric in FOLD_METRICS:
            rows.append(
                keys
                | {
                    "metric": metric,
                    "fold": int(fold_row.fold),
                    "value": float(getattr(fold_row, metric)),
                    "fell_back": fold_row.fell_back,
                }
            )
    flags = fold_metrics["fell_back"]
    any_fell_back = pd.NA if flags.isna().all() else bool(flags.fillna(False).any())
    aggregates = {name: metrics[key] for key, name in AGGREGATE_METRICS.items()}
    if params.get("min_lag") is None:
        aggregates["one_step_mae"] = metrics["mae_mean"]
    for metric, value in aggregates.items():
        rows.append(
            keys
            | {
                "metric": metric,
                "fold": pd.NA,
                "value": float(value),
                "fell_back": any_fell_back,
            }
        )
    out = pd.DataFrame(rows)
    out["fold"] = out["fold"].astype("Int16")
    out["fell_back"] = out["fell_back"].astype("boolean")
    return out


def replace_rows(conn: Connection, table: str, rows: pd.DataFrame) -> None:
    """Delete this (isbn, model_type, run_id)'s rows from table, then insert rows.

    Same run_id twice leaves the same rows, a new run_id appends. Call inside one
    transaction (engine.begin()) so a failed insert does not lose the old rows.
    """
    key_cols = TABLE_KEYS[table]
    key = {col: rows[col].iloc[0] for col in key_cols}
    for col in key_cols:
        if rows[col].nunique() != 1:
            raise ValueError(f"rows for {table} span more than one {col}")
    where = " AND ".join(f"{col} = :{col}" for col in key_cols)
    conn.execute(text(f"DELETE FROM {table} WHERE {where}"), key)
    rows.to_sql(table, conn, if_exists="append", index=False, method="multi")


def write_forecasts(
    predictions: pd.DataFrame, params: dict, connection_string: str
) -> str:
    """Kedro node: write the backtest predictions of the active MLflow run."""
    rows = build_forecast_rows(predictions, params, current_run_id())
    engine = create_engine(connection_string)
    try:
        with engine.begin() as conn:
            replace_rows(conn, "forecasts", rows)
    finally:
        engine.dispose()
    logger.info("write_forecasts: %d rows for %s/%s", len(rows), rows["isbn"].iloc[0], rows["model_type"].iloc[0])
    return "forecasts_written"


def write_model_metrics(
    metrics: dict,
    fold_metrics: pd.DataFrame,
    params: dict,
    connection_string: str,
    _forecasts_written: str,
) -> str:
    """Kedro node: write the metrics of the active MLflow run."""
    rows = build_metric_rows(metrics, fold_metrics, params, current_run_id())
    engine = create_engine(connection_string)
    try:
        with engine.begin() as conn:
            replace_rows(conn, "model_metrics", rows)
    finally:
        engine.dispose()
    logger.info("write_model_metrics: %d rows for %s/%s", len(rows), rows["isbn"].iloc[0], rows["model_type"].iloc[0])
    return "model_metrics_written"