import logging
import os
from pathlib import Path

import pandas as pd
import pytest

from nielsen_scope.pipelines.training import nodes
from nielsen_scope.pipelines.training.nodes import (
    cross_validate,
    select_book_features,
)

logger = logging.getLogger(__name__)

ALCHEMIST = "9780722532935"
N_SPOT_CHECKS = 3
RUN_SLOW = bool(os.environ.get("RUN_SLOW"))
PARQUET = Path(__file__).resolve().parents[3] / "data/04_feature/sales_features.parquet"
RAW_XLSX = (
    Path(__file__).resolve().parents[3]
    / "data/01_raw/UK Weekly Trended Timeline from 200101_202429.xlsx"
)
pytestmark = pytest.mark.skipif(not PARQUET.exists(), reason="feature parquet not built")


@pytest.fixture(scope="module")
def features() -> pd.DataFrame:
    return pd.read_parquet(PARQUET)


@pytest.fixture(scope="module")
def raw_volume() -> pd.Series:
    """Raw weekly Volume for the Alchemist, indexed by the raw (Saturday) End Date."""
    raw = pd.concat(pd.read_excel(RAW_XLSX, sheet_name=None).values())
    raw = raw[raw["ISBN"].astype(str) == ALCHEMIST]
    return raw.set_index("End Date")["Volume"].sort_index()


def _params(**overrides) -> dict:
    return {
        "isbn": ALCHEMIST,
        "model_start_date": "2012-01-01",
        "model_type": "xgboost",
        "min_lag": None,
        "exclude_features": [],
        "cv": {"n_splits": 5, "test_horizon": 32},
        "model_params": {
            "xgboost": {
                "n_estimators": 100,
                "max_depth": 7,
                "learning_rate": 0.2,
                "objective": "reg:squarederror",
                "random_state": 42,
            }
        },
    } | overrides


def _naive_params(**overrides) -> dict:
    return _params(model_type="naive", model_params={}, **overrides)


def _crostonsba_params() -> dict:
    return _params(model_type="crostonsba", model_params={"crostonsba": {}})


def _autoarima_params() -> dict:
    autoarima = {
        "season_length": 52,
        "max_p": 5,
        "max_q": 5,
        "max_P": 2,
        "max_Q": 2,
        "stepwise": True,
        "ic": "aic",
        "fit_timeout_seconds": 3600,
    }
    return _params(model_type="autoarima", model_params={"autoarima": autoarima})


@pytest.fixture(autouse=True)
def _silence_mlflow(monkeypatch):
    monkeypatch.setattr(nodes.mlflow, "log_metric", lambda *a, **k: None)
    monkeypatch.setattr(nodes.mlflow, "log_metrics", lambda *a, **k: None)


def test_legacy_features_reproduce_stage2_one_step_mae(features):
    params = _params()
    X, y, dates = select_book_features(features, params)
    metrics, _ = cross_validate(X, y, params, dates)
    assert metrics["mae_mean"] == pytest.approx(92.9642066955566, abs=1e-4)


def test_direct_features_reproduce_year_ablation_mae(features):
    params = _params(min_lag=32, exclude_features=["year"])
    X, y, dates = select_book_features(features, params)
    assert list(X.columns) == [
        "volume_lag_52", "week_of_year", "month", "quarter", "rrp",
    ]
    metrics, _ = cross_validate(X, y, params, dates)
    assert metrics["mae_mean"] == pytest.approx(174.349, abs=5e-4)


def test_naive_reproduces_planned_baseline_mae(features):
    params = _naive_params()
    X, y, dates = select_book_features(features, params)
    metrics, _ = cross_validate(X, y, params, dates)
    assert metrics["mae_mean"] == pytest.approx(167.519, abs=5e-4)
    assert metrics["relative_mae"] == pytest.approx(1.0)
    assert metrics["mape_excluded_rows"] == 0


def test_naive_mae_is_unchanged_by_min_lag(features):
    params = _naive_params(min_lag=32, exclude_features=["year"])
    X, y, dates = select_book_features(features, params)
    metrics, _ = cross_validate(X, y, params, dates)
    assert metrics["mae_mean"] == pytest.approx(167.519, abs=5e-4)


def test_xgboost_relative_mae_against_naive(features):
    params = _params()
    X, y, dates = select_book_features(features, params)
    metrics, _ = cross_validate(X, y, params, dates)
    assert metrics["relative_mae"] == pytest.approx(0.55495, abs=5e-5)


@pytest.mark.skipif(not RAW_XLSX.exists(), reason="raw xlsx not present")
def test_backtest_rows_match_raw_source(features, raw_volume):
    params = _naive_params()
    X, y, dates = select_book_features(features, params)
    _, preds = cross_validate(X, y, params, dates)
    sample = preds[preds["y_true"] > 0].sample(n=N_SPOT_CHECKS, random_state=42)
    checked = pd.concat([sample, preds.iloc[[-1]]])
    # stored end_date is the Sunday after the raw Saturday week-end (A-005)
    saturday = pd.Timedelta(days=1)
    year = pd.Timedelta(weeks=52)
    for row in checked.itertuples():
        assert row.y_true == raw_volume.loc[row.end_date - saturday]
        assert row.y_pred == raw_volume.get(row.end_date - year - saturday, 0)
    logger.info(
        "spot-check rows:\n%s", checked[["end_date", "y_true", "y_pred"]].to_string()
    )


def test_crostonsba_reproduces_spike_mae(features):
    params = _crostonsba_params()
    X, y, dates = select_book_features(features, params)
    metrics, _ = cross_validate(X, y, params, dates)
    assert metrics["mae_mean"] == pytest.approx(165.041, abs=5e-4)
    assert metrics["relative_mae"] == pytest.approx(0.985209, abs=1e-6)


@pytest.mark.skipif(not RUN_SLOW, reason="set RUN_SLOW=1: five AutoARIMA fits, ~25 min")
def test_autoarima_reproduces_spike_mae(features):
    params = _autoarima_params()
    X, y, dates = select_book_features(features, params)
    metrics, _ = cross_validate(X, y, params, dates)
    assert metrics["mae_mean"] == pytest.approx(145.3503, abs=5e-4)
    assert metrics["relative_mae"] == pytest.approx(0.867666, abs=1e-6)
