import numpy as np
import pandas as pd
import pytest

from nielsen_scope.pipelines.training import nodes
from nielsen_scope.pipelines.training.nodes import cross_validate, select_book_features

ISBN = "9780722532935"
N_FOLDS = 5
FOLD_COLUMNS = [
    "fold", "mae", "rmse", "mape", "mape_excluded_rows",
    "naive_mae", "rel_mae", "fell_back",
]
AUTOARIMA = {
    "season_length": 4,
    "max_p": 2,
    "max_q": 2,
    "max_P": 1,
    "max_Q": 1,
    "stepwise": True,
    "ic": "aic",
    "fit_timeout_seconds": 600,
}


@pytest.fixture(autouse=True)
def _silence_mlflow(monkeypatch):
    monkeypatch.setattr(nodes.mlflow, "log_metric", lambda *a, **k: None)
    monkeypatch.setattr(nodes.mlflow, "log_metrics", lambda *a, **k: None)


def _features(n: int = 400) -> pd.DataFrame:
    dates = pd.date_range("2012-01-01", periods=n, freq="W")
    rng = np.random.default_rng(42)
    vol = (50 + 10 * np.sin(np.arange(n) * np.pi / 2) + rng.integers(0, 5, n)).astype("int64")
    return pd.DataFrame(
        {
            "isbn": ISBN,
            "end_date": dates.date,
            "volume": vol,
            "volume_lag_52": pd.Series(vol).shift(52).to_numpy(),
            "week_of_year": dates.isocalendar().week.to_numpy().astype("int64"),
        }
    )


def _params(model_type: str, model_params: dict) -> dict:
    return {
        "isbn": ISBN,
        "model_start_date": "2012-01-01",
        "model_type": model_type,
        "min_lag": None,
        "exclude_features": [],
        "cv": {"n_splits": N_FOLDS, "test_horizon": 32},
        "model_params": {model_type: model_params},
    }


def _run(model_type: str, model_params: dict):
    params = _params(model_type, model_params)
    X, y, dates = select_book_features(_features(), params)
    return cross_validate(X, y, params, dates)


def test_xgboost_fold_metrics_match_aggregates_and_have_no_intervals():
    metrics, preds, fold_metrics = _run("xgboost", {"n_estimators": 20, "random_state": 42})
    assert list(fold_metrics.columns) == FOLD_COLUMNS
    assert list(fold_metrics["fold"]) == list(range(N_FOLDS))
    assert metrics["mae_mean"] == pytest.approx(fold_metrics["mae"].mean())
    assert fold_metrics["fell_back"].isna().all()
    assert preds["lo_95"].isna().all() and preds["hi_95"].isna().all()


def test_naive_fold_relative_mae_is_one():
    _, _, fold_metrics = _run("naive", {})
    assert fold_metrics["rel_mae"].to_numpy() == pytest.approx(1.0)
    assert fold_metrics["naive_mae"].to_numpy() == pytest.approx(fold_metrics["mae"].to_numpy())


def test_autoarima_returns_intervals_around_the_forecast():
    _, preds, fold_metrics = _run("autoarima", AUTOARIMA)
    assert np.isfinite(preds[["lo_95", "hi_95"]].to_numpy()).all()
    assert (preds["lo_95"] <= preds["y_pred"]).all()
    assert (preds["y_pred"] <= preds["hi_95"]).all()
    assert not fold_metrics["fell_back"].any()


def test_autoarima_fallback_is_reported_per_fold():
    _, preds, fold_metrics = _run("autoarima", AUTOARIMA | {"fit_timeout_seconds": 0.01})
    assert fold_metrics["fell_back"].all()
    assert np.isfinite(preds[["lo_95", "hi_95"]].to_numpy()).all()
