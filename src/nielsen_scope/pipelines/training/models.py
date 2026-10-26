"""Model dispatch for the walk-forward evaluator: model_type -> (fit, predict)."""

import logging
import multiprocessing
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, CrostonSBA
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)

STATS_FREQ = "W-SUN"
STATS_START = "2000-01-02"
STATS_UNIQUE_ID = "series"


def fit_xgboost(X: pd.DataFrame, y: pd.Series, model_params: dict) -> XGBRegressor:
    model = XGBRegressor(**model_params)
    model.fit(X, y)
    return model


def predict_xgboost(model: XGBRegressor, X: pd.DataFrame) -> np.ndarray:
    return model.predict(X)


def fit_catboost(X: pd.DataFrame, y: pd.Series, model_params: dict) -> CatBoostRegressor:
    """allow_writing_files defaults to False so a fit does not write catboost_info/
    into the working directory; model_params can override it."""
    model = CatBoostRegressor(**{"allow_writing_files": False, **model_params})
    model.fit(X, y)
    return model


def predict_catboost(model: CatBoostRegressor, X: pd.DataFrame) -> np.ndarray:
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


@dataclass
class FittedStatsForecast:
    """A fitted StatsForecast object plus what predict needs to read it back."""

    sf: StatsForecast
    column: str
    fell_back: bool


def _build_statsforecast(model_name: str, model_params: dict) -> StatsForecast:
    model_classes = {
        "AutoARIMA": AutoARIMA,
        "AutoETS": AutoETS,
        "CrostonSBA": CrostonSBA,
    }
    model = model_classes[model_name](**model_params)
    return StatsForecast(models=[model], freq=STATS_FREQ)


def _fit_worker(model_name: str, model_params: dict, frame: pd.DataFrame, conn) -> None:
    sf = _build_statsforecast(model_name, model_params)
    sf.fit(frame)
    conn.send(sf)
    conn.close()


def _fit_with_cap(
    model_name: str, model_params: dict, frame: pd.DataFrame, timeout: float | None
) -> StatsForecast | None:
    """Fit in a spawned subprocess and kill it after timeout seconds.

    statsforecast models have no timeout argument, so the cap has to be external.
    Returns None when the fit timed out or the subprocess died; timeout=None
    waits for ever.
    """
    ctx = multiprocessing.get_context("spawn")
    receiver, sender = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_fit_worker, args=(model_name, model_params, frame, sender)
    )
    proc.start()
    sender.close()
    fitted = None
    try:
        if receiver.poll(timeout):
            fitted = receiver.recv()
    except EOFError:
        pass
    if fitted is None:
        proc.terminate()
    proc.join()
    receiver.close()
    return fitted


def _fit_statsforecast(
    model_name: str, model_params: dict, y: pd.Series, timeout: float | None
) -> FittedStatsForecast:
    """Fit one statsforecast model on y (X is ignored); fall back to AutoETS if the
    capped fit did not finish. The dates are a regular weekly grid: forecasts here
    are positional, and season_length is passed explicitly."""
    frame = pd.DataFrame(
        {
            "unique_id": STATS_UNIQUE_ID,
            "ds": pd.date_range(STATS_START, periods=len(y), freq=STATS_FREQ),
            "y": y.to_numpy(dtype=float),
        }
    )
    start = time.perf_counter()
    sf = _fit_with_cap(model_name, model_params, frame, timeout)
    elapsed = time.perf_counter() - start
    fell_back = sf is None
    if fell_back:
        logger.warning(
            "%s fit did not finish within %ss (%.1fs elapsed) or failed; "
            "falling back to AutoETS",
            model_name, timeout, elapsed,
        )
        sf = _build_statsforecast(
            "AutoETS", {"season_length": model_params.get("season_length", 1)}
        )
        sf.fit(frame)
    else:
        logger.info("%s fit: %.1fs on %d rows", model_name, elapsed, len(y))
    return FittedStatsForecast(sf=sf, column=sf.models[0].alias, fell_back=fell_back)


def fit_autoarima(X: pd.DataFrame, y: pd.Series, model_params: dict) -> FittedStatsForecast:
    params = dict(model_params)
    timeout = params.pop("fit_timeout_seconds", None)
    return _fit_statsforecast("AutoARIMA", params, y, timeout)


def fit_crostonsba(X: pd.DataFrame, y: pd.Series, model_params: dict) -> FittedStatsForecast:
    return _fit_statsforecast("CrostonSBA", dict(model_params), y, None)


def predict_statsforecast(model: FittedStatsForecast, X: pd.DataFrame) -> np.ndarray:
    """Point forecast for the next len(X) weeks."""
    forecast = model.sf.predict(h=len(X))
    return forecast[model.column].to_numpy()


MODELS: dict[str, tuple[Callable, Callable]] = {
    "xgboost": (fit_xgboost, predict_xgboost),
    "catboost": (fit_catboost, predict_catboost),
    "naive": (fit_naive, predict_naive),
    "autoarima": (fit_autoarima, predict_statsforecast),
    "crostonsba": (fit_crostonsba, predict_statsforecast),
}


def get_model(model_type: str) -> tuple[Callable, Callable]:
    """Return the (fit, predict) pair registered for model_type."""
    if model_type not in MODELS:
        raise ValueError(
            f"Unknown model_type {model_type!r}; registered: {sorted(MODELS)}"
        )
    return MODELS[model_type]
