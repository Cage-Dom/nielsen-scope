"""Model dispatch for the walk-forward evaluator: model_type -> (fit, predict)."""

from collections.abc import Callable

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


def fit_xgboost(X: pd.DataFrame, y: pd.Series, model_params: dict) -> XGBRegressor:
    model = XGBRegressor(**model_params)
    model.fit(X, y)
    return model


def predict_xgboost(model: XGBRegressor, X: pd.DataFrame) -> np.ndarray:
    return model.predict(X)


NAIVE_LAG_COLUMN = "volume_lag_52"


def fit_naive(X: pd.DataFrame, y: pd.Series, model_params: dict) -> None:
    return None


def predict_naive(model: None, X: pd.DataFrame) -> np.ndarray:
    """Seasonal naive: the forecast for a week is the volume 52 weeks earlier."""
    if NAIVE_LAG_COLUMN not in X.columns:
        raise ValueError(
            f"naive needs {NAIVE_LAG_COLUMN}; it was excluded or dropped by min_lag"
        )
    pred = X[NAIVE_LAG_COLUMN].to_numpy(dtype=float)
    if np.isnan(pred).any():
        raise ValueError(f"{NAIVE_LAG_COLUMN} has NaN in the rows being forecast")
    return pred


MODELS: dict[str, tuple[Callable, Callable]] = {
    "xgboost": (fit_xgboost, predict_xgboost),
    "naive": (fit_naive, predict_naive),
}


def get_model(model_type: str) -> tuple[Callable, Callable]:
    """Return the (fit, predict) pair registered for model_type."""
    if model_type not in MODELS:
        raise ValueError(
            f"Unknown model_type {model_type!r}; registered: {sorted(MODELS)}"
        )
    return MODELS[model_type]
