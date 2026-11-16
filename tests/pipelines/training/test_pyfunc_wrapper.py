import mlflow
import numpy as np
import pandas as pd
import pytest

from nielsen_scope.pipelines.training.models import (
    FittedStatsForecast,
    fit_autoarima,
    fit_crostonsba,
)
from nielsen_scope.pipelines.training.pyfunc import log_statsforecast_model

HORIZON = 32
SHORT_HORIZON = 7
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
def temp_mlflow_store(tmp_path):
    """Log to a throwaway store and artifact folder, never the project's mlflow.db or mlruns."""
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    experiment_id = mlflow.create_experiment(
        "pyfunc-test", artifact_location=(tmp_path / "artifacts").as_uri()
    )
    mlflow.set_experiment(experiment_id=experiment_id)
    yield
    mlflow.set_tracking_uri(previous)


@pytest.fixture(scope="module")
def series() -> pd.Series:
    rng = np.random.default_rng(42)
    n = 200
    return pd.Series(50 + 10 * np.sin(np.arange(n) * np.pi / 2) + rng.integers(0, 5, n))


@pytest.fixture(scope="module")
def autoarima(series) -> FittedStatsForecast:
    return fit_autoarima(pd.DataFrame(), series, AUTOARIMA)


@pytest.fixture(scope="module")
def crostonsba(series) -> FittedStatsForecast:
    return fit_crostonsba(pd.DataFrame(), series, {})


def _roundtrip(fitted, level=None):
    with mlflow.start_run():
        info = log_statsforecast_model(fitted, "model", level=level)
    return mlflow.pyfunc.load_model(info.model_uri), info


@pytest.mark.parametrize("fixture_name", ["autoarima", "crostonsba"])
def test_reloaded_forecast_equals_the_original(fixture_name, request):
    fitted = request.getfixturevalue(fixture_name)
    loaded, _ = _roundtrip(fitted)
    got = loaded.predict(pd.DataFrame(index=range(HORIZON)))
    want = fitted.sf.predict(h=HORIZON).reset_index()
    assert list(got.columns) == list(want.columns)
    assert got["ds"].tolist() == want["ds"].tolist()
    np.testing.assert_allclose(
        got[fitted.column].to_numpy(), want[fitted.column].to_numpy(), rtol=0, atol=1e-9
    )


def test_the_horizon_is_the_number_of_input_rows(autoarima):
    loaded, _ = _roundtrip(autoarima)
    assert len(loaded.predict(pd.DataFrame(index=range(SHORT_HORIZON)))) == SHORT_HORIZON


def test_interval_columns_match_the_original_when_a_level_is_logged(autoarima):
    loaded, _ = _roundtrip(autoarima, level=95)
    got = loaded.predict(pd.DataFrame(index=range(HORIZON)))
    want = autoarima.sf.predict(h=HORIZON, level=[95]).reset_index()
    for column in (f"{autoarima.column}-lo-95", f"{autoarima.column}-hi-95"):
        np.testing.assert_allclose(
            got[column].to_numpy(), want[column].to_numpy(), rtol=0, atol=1e-9
        )


def test_fell_back_flag_is_kept_in_the_model_metadata(crostonsba):
    flagged = FittedStatsForecast(sf=crostonsba.sf, column=crostonsba.column, fell_back=True)
    _, info = _roundtrip(flagged)
    assert mlflow.models.get_model_info(info.model_uri).metadata["fell_back"] is True


def test_models_are_stored_under_the_temp_folder(tmp_path, crostonsba):
    with mlflow.start_run():
        log_statsforecast_model(crostonsba, "model")
        assert mlflow.get_artifact_uri().startswith((tmp_path / "artifacts").as_uri())
