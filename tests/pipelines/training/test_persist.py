import os
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import IntegrityError, OperationalError

from nielsen_scope.pipelines.training import persist
from nielsen_scope.pipelines.training.persist import (
    build_forecast_rows,
    build_metric_rows,
    current_run_id,
    replace_rows,
)

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_SQL = ROOT / "src/nielsen_scope/sql/schema.sql"
ISBN = "9780722532935"
RUN_A = "a" * 32
RUN_B = "b" * 32
N_FOLDS = 5
HORIZON = 32
N_FORECAST_ROWS = N_FOLDS * HORIZON
N_FOLD_METRICS = 5
N_AGGREGATES = 8
N_METRIC_ROWS = N_FOLDS * N_FOLD_METRICS + N_AGGREGATES + 1
FALLBACK_FOLD = 2
# the statement ingestion.nodes.truncate_tables runs on every full load
INGESTION_TRUNCATE = "TRUNCATE TABLE sales, books, categories RESTART IDENTITY CASCADE"


def _params(**overrides) -> dict:
    return {
        "isbn": ISBN,
        "model_type": "autoarima",
        "min_lag": None,
        "cv": {"n_splits": N_FOLDS, "test_horizon": HORIZON},
    } | overrides


def _predictions() -> pd.DataFrame:
    dates = pd.date_range("2024-01-07", periods=N_FORECAST_ROWS, freq="W")
    y_true = np.arange(N_FORECAST_ROWS)
    y_true[0] = -1  # a negative actual is accepted on purpose (A-070)
    return pd.DataFrame(
        {
            "fold": np.repeat(np.arange(N_FOLDS), HORIZON),
            "end_date": dates,
            "y_true": y_true,
            "y_pred": y_true + 0.5,
            "lo_95": y_true - 10.0,
            "hi_95": y_true + 11.0,
        }
    )


def _fold_metrics(flags=None) -> pd.DataFrame:
    flags = [pd.NA] * N_FOLDS if flags is None else flags
    return pd.DataFrame(
        {
            "fold": range(N_FOLDS),
            "mae": [10.0 + i for i in range(N_FOLDS)],
            "rmse": [12.0 + i for i in range(N_FOLDS)],
            "mape": [0.5] * N_FOLDS,
            "mape_excluded_rows": [0] * N_FOLDS,
            "naive_mae": [20.0] * N_FOLDS,
            "rel_mae": [0.5 + 0.1 * i for i in range(N_FOLDS)],
            "fell_back": pd.array(flags, dtype="boolean"),
        }
    )


def _metrics(**overrides) -> dict:
    return {
        "mae_mean": 12.0,
        "mae_std": 1.4,
        "rmse_mean": 14.0,
        "rmse_std": 1.4,
        "mape_mean": 0.5,
        "mape_std": 0.0,
        "relative_mae": 0.7,
        "mape_excluded_rows": 0,
        "n_folds": N_FOLDS,
    } | overrides


def test_forecast_rows_have_the_table_shape():
    rows = build_forecast_rows(_predictions(), _params(), RUN_A)
    assert len(rows) == N_FORECAST_ROWS
    assert set(rows["kind"]) == {"backtest"}
    assert set(rows["data_version"]) == {"nielsen_raw"}
    assert set(rows["run_id"]) == {RUN_A}
    assert rows["end_date"].is_unique
    assert rows["y_true"].iloc[0] == -1


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda p: p.iloc[:-1], "expected 160"),
        (lambda p: p.assign(end_date=p["end_date"].iloc[0]), "not unique"),
        (lambda p: p.assign(end_date=pd.NaT), "need end_date"),
        (lambda p: p.assign(y_pred=np.nan), "NaN or infinite"),
        (lambda p: p.assign(y_true=p["y_true"] + 0.5), "non-integer"),
    ],
)
def test_forecast_rows_reject_bad_predictions(change, message):
    with pytest.raises(ValueError, match=message):
        build_forecast_rows(change(_predictions()), _params(), RUN_A)


def test_metric_rows_cover_folds_and_aggregates():
    rows = build_metric_rows(_metrics(), _fold_metrics(), _params(), RUN_A)
    assert len(rows) == N_METRIC_ROWS
    folds = rows[rows["fold"].notna()]
    aggregates = rows[rows["fold"].isna()]
    assert len(folds) == N_FOLDS * N_FOLD_METRICS
    assert set(aggregates["metric"]) == {
        "mae", "mae_std", "rmse", "rmse_std", "mape", "mape_std",
        "rel_mae", "mape_excluded_rows", "one_step_mae",
    }
    assert aggregates.set_index("metric").loc["rel_mae", "value"] == pytest.approx(0.7)


def test_metric_rows_skip_one_step_mae_when_min_lag_is_set():
    rows = build_metric_rows(_metrics(), _fold_metrics(), _params(min_lag=32), RUN_A)
    assert len(rows) == N_METRIC_ROWS - 1
    assert "one_step_mae" not in set(rows["metric"])


def test_metric_rows_keep_an_undefined_metric_as_a_null_value():
    rows = build_metric_rows(
        _metrics(relative_mae=float("nan")), _fold_metrics(), _params(), RUN_A
    )
    rel = rows[(rows["metric"] == "rel_mae") & rows["fold"].isna()]
    assert len(rel) == 1 and rel["value"].isna().all()


def test_metric_rows_carry_the_fallback_flag():
    flags = [False] * N_FOLDS
    flags[FALLBACK_FOLD] = True
    rows = build_metric_rows(_metrics(), _fold_metrics(flags), _params(), RUN_A)
    flagged = rows[rows["fold"] == FALLBACK_FOLD]
    assert flagged["fell_back"].all()
    assert not rows[rows["fold"] == 0]["fell_back"].any()
    assert rows[rows["fold"].isna()]["fell_back"].all()  # any fold fell back


def test_metric_rows_leave_the_flag_null_for_models_that_cannot_fall_back():
    rows = build_metric_rows(_metrics(), _fold_metrics(), _params(), RUN_A)
    assert rows["fell_back"].isna().all()


def test_current_run_id_needs_an_active_run(monkeypatch):
    monkeypatch.setattr(persist.mlflow, "active_run", lambda: None)
    with pytest.raises(RuntimeError, match="no active MLflow run"):
        current_run_id()


def _test_db_url() -> str:
    """The disposable test database; never the project's own (conf/local) database."""
    url = os.environ.get("NIELSEN_TEST_DB_URL")
    if not url:
        pytest.skip("set NIELSEN_TEST_DB_URL to a disposable database to run these")
    database = make_url(url).database or ""
    if not database.endswith("_test"):
        pytest.fail(f"refusing to use database {database!r}: its name must end in _test")
    return url


@pytest.fixture
def db():
    """An engine whose search_path is a throwaway schema holding the project tables."""
    url = _test_db_url()
    schema = f"t23_{uuid.uuid4().hex[:12]}"
    admin = create_engine(url)
    try:
        with admin.begin() as conn:
            conn.execute(text(f"CREATE SCHEMA {schema}"))
    except OperationalError:
        admin.dispose()
        pytest.skip("postgres not reachable")
    engine = create_engine(url, connect_args={"options": f"-csearch_path={schema}"})
    with engine.begin() as conn:
        for statement in SCHEMA_SQL.read_text().split(";"):
            if statement.strip():
                conn.execute(text(statement))
    yield engine
    engine.dispose()
    with admin.begin() as conn:
        conn.execute(text(f"DROP SCHEMA {schema} CASCADE"))
    admin.dispose()


def _count(engine, table: str) -> int:
    with engine.connect() as conn:
        return conn.execute(text(f"SELECT count(*) FROM {table}")).scalar_one()


def _write_forecasts(engine, run_id: str) -> None:
    rows = build_forecast_rows(_predictions(), _params(), run_id)
    with engine.begin() as conn:
        replace_rows(conn, "forecasts", rows)


def _write_metrics(engine, run_id: str, flags=None) -> None:
    rows = build_metric_rows(_metrics(), _fold_metrics(flags), _params(), run_id)
    with engine.begin() as conn:
        replace_rows(conn, "model_metrics", rows)


def test_rewriting_the_same_run_keeps_row_counts(db):
    _write_forecasts(db, RUN_A)
    _write_forecasts(db, RUN_A)
    _write_metrics(db, RUN_A)
    _write_metrics(db, RUN_A)
    assert _count(db, "forecasts") == N_FORECAST_ROWS
    assert _count(db, "model_metrics") == N_METRIC_ROWS


def test_a_new_run_appends_so_history_is_kept(db):
    _write_forecasts(db, RUN_A)
    _write_forecasts(db, RUN_B)
    _write_metrics(db, RUN_A)
    _write_metrics(db, RUN_B)
    assert _count(db, "forecasts") == 2 * N_FORECAST_ROWS
    assert _count(db, "model_metrics") == 2 * N_METRIC_ROWS


def test_intervals_and_negative_actuals_round_trip(db):
    _write_forecasts(db, RUN_A)
    with db.connect() as conn:
        row = conn.execute(
            text("SELECT y_true, y_pred, lo_95, hi_95 FROM forecasts ORDER BY end_date LIMIT 1")
        ).one()
    assert row == (-1, -0.5, -11.0, 10.0)


def test_nan_intervals_are_stored_as_null(db):
    predictions = _predictions().assign(lo_95=np.nan, hi_95=np.nan)
    rows = build_forecast_rows(predictions, _params(model_type="xgboost"), RUN_A)
    with db.begin() as conn:
        replace_rows(conn, "forecasts", rows)
    with db.connect() as conn:
        nulls = conn.execute(
            text("SELECT count(*) FROM forecasts WHERE lo_95 IS NULL AND hi_95 IS NULL")
        ).scalar_one()
    assert nulls == N_FORECAST_ROWS


def test_fallback_flag_round_trips(db):
    flags = [False] * N_FOLDS
    flags[FALLBACK_FOLD] = True
    _write_metrics(db, RUN_A, flags)
    with db.connect() as conn:
        fold_flags = dict(
            conn.execute(
                text("SELECT fold, bool_or(fell_back) FROM model_metrics WHERE fold IS NOT NULL GROUP BY fold")
            ).all()
        )
        aggregate = conn.execute(
            text("SELECT DISTINCT fell_back FROM model_metrics WHERE fold IS NULL")
        ).scalar_one()
    assert fold_flags[FALLBACK_FOLD] is True and fold_flags[0] is False
    assert aggregate is True


def _insert_forecast(conn, **values) -> None:
    row = {
        "isbn": ISBN, "model_type": "xgboost", "run_id": RUN_A, "kind": "backtest",
        "fold": 0, "end_date": "2024-01-07", "y_true": 1, "y_pred": 1.0,
    } | values
    conn.execute(
        text(
            "INSERT INTO forecasts (isbn, model_type, run_id, kind, fold, end_date, y_true, y_pred) "
            "VALUES (:isbn, :model_type, :run_id, :kind, :fold, :end_date, :y_true, :y_pred)"
        ),
        row,
    )


@pytest.mark.parametrize(
    "values",
    [
        {"fold": None},
        {"y_true": None},
        {"kind": "future"},
        {"kind": "future", "fold": None},
        {"kind": "other"},
        {"model_type": "lstm"},
    ],
)
def test_database_rejects_inconsistent_forecast_rows(db, values):
    with pytest.raises(IntegrityError), db.begin() as conn:
        _insert_forecast(conn, **values)


def test_database_accepts_a_future_row_without_fold_or_actual(db):
    with db.begin() as conn:
        _insert_forecast(conn, kind="future", fold=None, y_true=None)
    assert _count(db, "forecasts") == 1


def test_database_rejects_a_duplicate_aggregate_metric(db):
    insert = text(
        "INSERT INTO model_metrics (isbn, model_type, run_id, metric, fold, value) "
        "VALUES (:isbn, 'xgboost', :run_id, 'mae', NULL, 1.0)"
    )
    with db.begin() as conn:
        conn.execute(insert, {"isbn": ISBN, "run_id": RUN_A})
    with pytest.raises(IntegrityError), db.begin() as conn:
        conn.execute(insert, {"isbn": ISBN, "run_id": RUN_A})


def test_database_rejects_an_unknown_metric(db):
    with pytest.raises(IntegrityError), db.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO model_metrics (isbn, model_type, run_id, metric, fold, value) "
                "VALUES (:isbn, 'xgboost', :run_id, 'maee', NULL, 1.0)"
            ),
            {"isbn": ISBN, "run_id": RUN_A},
        )


def test_history_survives_the_ingestion_truncate(db):
    _write_forecasts(db, RUN_A)
    _write_metrics(db, RUN_A)
    with db.begin() as conn:
        conn.execute(text(INGESTION_TRUNCATE))
    assert _count(db, "forecasts") == N_FORECAST_ROWS
    assert _count(db, "model_metrics") == N_METRIC_ROWS
