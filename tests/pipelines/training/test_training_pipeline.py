import numpy as np
import pandas as pd
from xgboost import XGBRegressor

import nielsen_scope.pipelines.training.nodes as nodes
from nielsen_scope.pipelines.training.nodes import (
    select_book_features,
    train_final_xgb_model,
    walk_forward_evaluate_xgb,
)


def _synthetic_features(n: int = 400, isbn: str = "9780722532935") -> pd.DataFrame:
    dates = pd.date_range("2012-01-01", periods=n, freq='W')
    rng = np.random.default_rng(42)
    vol = np.arange(n) + rng.integers(0, 5, n)
    
    return pd.DataFrame(
        {
            "isbn": isbn,
            "end_date": dates.date,
            "volume": vol.astype("int64"),
            "volume_lag_1": pd.Series(vol).shift(1).to_numpy(),
            "week_of_year": dates.isocalendar().week.to_numpy().astype("int64"),
            "rrp": 7.99,
            "category": "F",
        }
    )


def _params() -> dict:
    return {
        "isbn": "9780722532935",
        "model_start_date": "2012-01-01",
        "model_type": "xgboost",
        "cv": {"n_splits": 5, "test_horizon": 32},
        "model_params": {"xgboost": {"n_estimators": 20, "max_depth": 3, "random_state": 42}},
    }


def test_select_book_features_drops_non_numeric_and_target():
    X, y = select_book_features(_synthetic_features(), _params())
    for dropped in ("volume", "isbn", "end_date", "category"):
        assert dropped not in X.columns
    assert list(X.index) == list(range(len(X)))  # contiguous RangeIndex
    assert y.name == "volume"


def test_walk_forward_evaluate_xgb_returns_finite_metrics(monkeypatch):
    monkeypatch.setattr(nodes.mlflow, "log_metric", lambda *a, **k: None)
    monkeypatch.setattr(nodes.mlflow, "log_metrics", lambda *a, **k: None)
    X, y = select_book_features(_synthetic_features(), _params())
    metrics = walk_forward_evaluate_xgb(X, y, _params())
    assert metrics["n_folds"] == 5
    assert np.isfinite(metrics["mae_mean"])
    assert np.isfinite(metrics["rmse_mean"])


def test_train_final_xgb_model_returns_fitted_estimator(monkeypatch):
    monkeypatch.setattr(nodes.mlflow, "log_params", lambda *a, **k: None)
    X, y = select_book_features(_synthetic_features(), _params())
    model = train_final_xgb_model(X, y, _params())
    assert isinstance(model, XGBRegressor)
    pred = model.predict(X)
    assert len(pred) == len(X) and np.all(np.isfinite(pred))