from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostRegressor

from nielsen_scope.pipelines.training import nodes
from nielsen_scope.pipelines.training.models import (
    fit_catboost,
    get_model,
    predict_catboost,
)
from nielsen_scope.pipelines.training.nodes import (
    cross_validate,
    select_book_features,
)

ALCHEMIST = "9780722532935"
PARQUET = Path(__file__).resolve().parents[3] / "data/04_feature/sales_features.parquet"
SMALL_PARAMS = {"iterations": 10, "random_seed": 42, "verbose": 0}
N_TAIL = 5


@pytest.fixture(autouse=True)
def _silence_mlflow(monkeypatch):
    monkeypatch.setattr(nodes.mlflow, "log_metric", lambda *a, **k: None)
    monkeypatch.setattr(nodes.mlflow, "log_metrics", lambda *a, **k: None)


def _xy(n: int = 60) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    return X, pd.Series(3 * X["a"] + rng.normal(size=n))


def _catboost_params(**overrides) -> dict:
    return {
        "isbn": ALCHEMIST,
        "model_start_date": "2012-01-01",
        "model_type": "catboost",
        "min_lag": None,
        "exclude_features": [],
        "cv": {"n_splits": 5, "test_horizon": 32},
        "model_params": {
            "catboost": {"random_seed": 42, "verbose": 0, "loss_function": "RMSE"}
        },
    } | overrides


@pytest.fixture(scope="module")
def features() -> pd.DataFrame:
    return pd.read_parquet(PARQUET)


def test_catboost_is_registered_and_predicts_without_writing_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    X, y = _xy()
    assert get_model("catboost") == (fit_catboost, predict_catboost)
    model = fit_catboost(X, y, SMALL_PARAMS)
    pred = predict_catboost(model, X.tail(N_TAIL))
    assert isinstance(model, CatBoostRegressor)
    assert len(pred) == N_TAIL and np.all(np.isfinite(pred))
    assert list(tmp_path.iterdir()) == []


def test_catboost_same_seed_gives_same_predictions():
    X, y = _xy()
    first = predict_catboost(fit_catboost(X, y, SMALL_PARAMS), X)
    second = predict_catboost(fit_catboost(X, y, SMALL_PARAMS), X)
    assert np.array_equal(first, second)


@pytest.mark.skipif(not PARQUET.exists(), reason="feature parquet not built")
def test_catboost_legacy_features_reproduce_spike_mae(features):
    params = _catboost_params()
    X, y, dates = select_book_features(features, params)
    metrics, _ = cross_validate(X, y, params, dates)
    assert metrics["mae_mean"] == pytest.approx(89.74814438, abs=5e-4)
    assert metrics["relative_mae"] == pytest.approx(0.535750, abs=1e-6)


@pytest.mark.skipif(not PARQUET.exists(), reason="feature parquet not built")
def test_catboost_direct_features_reproduce_spike_mae(features):
    params = _catboost_params(min_lag=32, exclude_features=["year"])
    X, y, dates = select_book_features(features, params)
    metrics, _ = cross_validate(X, y, params, dates)
    assert metrics["mae_mean"] == pytest.approx(163.32127515, abs=5e-4)
    assert metrics["relative_mae"] == pytest.approx(0.974943, abs=1e-6)