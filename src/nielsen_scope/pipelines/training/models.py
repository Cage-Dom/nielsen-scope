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


MODELS: dict[str, tuple[Callable, Callable]] = {
    "xgboost": (fit_xgboost, predict_xgboost),
}


def get_model(model_type: str) -> tuple[Callable, Callable]:
    """Return the (fit, predict) pair registered for model_type."""
    if model_type not in MODELS:
        raise ValueError(
            f"Unknown model_type {model_type!r}; registered: {sorted(MODELS)}"
        )
    return MODELS[model_type]
