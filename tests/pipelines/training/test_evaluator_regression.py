from pathlib import Path

import pandas as pd
import pytest

import nielsen_scope.pipelines.training.nodes as nodes
from nielsen_scope.pipelines.training.nodes import (
    cross_validate,
    select_book_features,
)

PARQUET = Path(__file__).resolve().parents[3] / "data/04_feature/sales_features.parquet"
pytestmark = pytest.mark.skipif(not PARQUET.exists(), reason="feature parquet not built")


@pytest.fixture(scope="module")
def features() -> pd.DataFrame:
    return pd.read_parquet(PARQUET)


def _params(**overrides) -> dict:
    return {
        "isbn": "9780722532935",
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