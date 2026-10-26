import numpy as np
import optuna
import pandas as pd
import pytest
from xgboost import XGBRegressor

import nielsen_scope.pipelines.training.nodes as nodes
from nielsen_scope.pipelines.training.nodes import (
    select_book_features,
    train_final_xgb_model,
    cross_validate,
    objective,
    optuna_hyperparam_tune_xgb,
)
from nielsen_scope.pipelines.training.models import get_model

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
    X, y, _ = select_book_features(_synthetic_features(), _params())
    for dropped in ("volume", "isbn", "end_date", "category"):
        assert dropped not in X.columns
    assert list(X.index) == list(range(len(X)))  # contiguous RangeIndex
    assert y.name == "volume"


def test_cross_validate_returns_finite_metrics_and_predictions(monkeypatch):
    monkeypatch.setattr(nodes.mlflow, "log_metric", lambda *a, **k: None)
    monkeypatch.setattr(nodes.mlflow, "log_metrics", lambda *a, **k: None)
    X, y, dates = select_book_features(_synthetic_features(), _params())
    metrics, preds = cross_validate(X, y, _params(), dates)
    assert metrics["n_folds"] == 5
    assert np.isfinite(metrics["mae_mean"])
    assert np.isfinite(metrics["rmse_mean"])
    assert list(preds.columns) == ["fold", "end_date", "y_true", "y_pred"]
    assert len(preds) == 160 and preds["end_date"].is_unique


def test_min_lag_drops_short_lags_and_rolls():
    params = _params() | {"min_lag": 32}
    X, _, _ = select_book_features(_synthetic_features(), params)
    assert "volume_lag_1" not in X.columns and "volume_roll_4" not in X.columns
    assert "volume_lag_52" in X.columns and "week_of_year" in X.columns


def test_min_lag_below_test_horizon_raises():
    with pytest.raises(ValueError, match="min_lag"):
        select_book_features(_synthetic_features(), _params() | {"min_lag": 16})


def test_exclude_features_drops_and_unknown_name_raises():
    X, _, _ = select_book_features(
        _synthetic_features(), _params() | {"exclude_features": ["year"]}
    )
    assert "year" not in X.columns
    with pytest.raises(ValueError, match="nope"):
        select_book_features(
            _synthetic_features(), _params() | {"exclude_features": ["nope"]}
        )


def test_get_model_unknown_type_raises():
    with pytest.raises(ValueError, match="Unknown model_type"):
        get_model("nope")


def test_train_final_xgb_model_returns_fitted_estimator(monkeypatch):
    monkeypatch.setattr(nodes.mlflow, "log_params", lambda *a, **k: None)
    X, y, _ = select_book_features(_synthetic_features(), _params())
    model = train_final_xgb_model(X, y, _params())
    assert isinstance(model, XGBRegressor)
    pred = model.predict(X)
    assert len(pred) == len(X) and np.all(np.isfinite(pred))


def _tuning_params() -> dict:
    params = _params()
    params["hyperparam_tuning"] = {
        "optuna": {"n_trials": 2, "n_warmup_steps": 5, "direction": "minimize"},
        "xgboost": {
            "n_estimators": {"lower": 5, "upper": 10},
            "max_depth": {"lower": 2, "upper": 3},
            "learning_rate": {"lower": 0.05, "upper": 0.2},
        },
    }
    return params


def _silence_mlflow(monkeypatch):
    monkeypatch.setattr(nodes.mlflow, "log_metric", lambda *a, **k: None)
    monkeypatch.setattr(nodes.mlflow, "log_metrics", lambda *a, **k: None)


def test_objective_returns_finite_mae(monkeypatch):
    _silence_mlflow(monkeypatch)
    X, y, _ = select_book_features(_synthetic_features(), _params())
    trial = optuna.trial.FixedTrial(
        {"n_estimators": 8, "max_depth": 3, "learning_rate": 0.1}
    )
    score = objective(trial, _tuning_params(), X, y, "xgboost")
    assert np.isfinite(score) and score >= 0


def test_objective_rejects_non_xgboost_model():
    X, y, _ = select_book_features(_synthetic_features(), _params())
    with pytest.raises(NotImplementedError):
        objective(optuna.trial.FixedTrial({}), _tuning_params(), X, y, "catboost")


def test_tuner_returns_best_params_within_bounds(monkeypatch):
    _silence_mlflow(monkeypatch)
    X, y, _ = select_book_features(_synthetic_features(), _params())
    best = optuna_hyperparam_tune_xgb(X, y, _tuning_params())
    assert set(best) == {"n_estimators", "max_depth", "learning_rate"}
    assert 5 <= best["n_estimators"] <= 10
    assert 2 <= best["max_depth"] <= 3
    assert 0.05 <= best["learning_rate"] <= 0.2