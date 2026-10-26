import logging

import numpy as np
import pandas as pd

from nielsen_scope.pipelines.training.models import get_model

HORIZON = 8
N_POINTS = 120
SMALL_ARIMA = {
    "season_length": 4,
    "max_p": 1,
    "max_q": 1,
    "max_P": 1,
    "max_Q": 1,
    "stepwise": True,
    "ic": "aic",
}


def _series() -> pd.Series:
    rng = np.random.default_rng(0)
    seasonal = 100 + 20 * np.sin(2 * np.pi * np.arange(N_POINTS) / 4)
    return pd.Series((seasonal + rng.normal(0, 3, N_POINTS)).round())


def _X(n: int) -> pd.DataFrame:
    return pd.DataFrame(index=range(n))


def test_crostonsba_predicts_one_value_per_test_week():
    fit, predict = get_model("crostonsba")
    model = fit(_X(N_POINTS), _series(), {})
    pred = predict(model, _X(HORIZON))
    assert len(pred) == HORIZON and np.all(np.isfinite(pred))
    assert model.fell_back is False


def test_autoarima_fits_in_a_subprocess_and_predicts():
    fit, predict = get_model("autoarima")
    params = SMALL_ARIMA | {"fit_timeout_seconds": 300}
    model = fit(_X(N_POINTS), _series(), params)
    pred = predict(model, _X(HORIZON))
    assert len(pred) == HORIZON and np.all(np.isfinite(pred))
    assert model.fell_back is False
    assert model.column == "AutoARIMA"


def test_autoarima_over_the_cap_falls_back_to_autoets(caplog):
    fit, predict = get_model("autoarima")
    params = SMALL_ARIMA | {"fit_timeout_seconds": 0.01}
    with caplog.at_level(logging.WARNING):
        model = fit(_X(N_POINTS), _series(), params)
    assert model.fell_back is True
    assert model.column == "AutoETS"
    assert "falling back to AutoETS" in caplog.text
    pred = predict(model, _X(HORIZON))
    assert len(pred) == HORIZON and np.all(np.isfinite(pred))
