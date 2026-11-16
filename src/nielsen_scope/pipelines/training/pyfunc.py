"""MLflow pyfunc wrapper for fitted statsforecast models (MLflow has no flavour for them)."""

import tempfile
from pathlib import Path

import mlflow
from mlflow.models.model import ModelInfo
from statsforecast import StatsForecast

from .models import FittedStatsForecast

ARTIFACT_KEY = "statsforecast"


class StatsForecastModel(mlflow.pyfunc.PythonModel):
    """Loads a saved StatsForecast; predict returns its forecast frame.

    The horizon is the number of rows in model_input, as in models.predict_statsforecast.
    Prediction intervals are included when the model was logged with a level.
    """

    def load_context(self, context):
        self.sf = StatsForecast.load(context.artifacts[ARTIFACT_KEY])
        self.level = (context.model_config or {}).get("level")

    def predict(self, context, model_input, params=None):
        forecast = self.sf.predict(h=len(model_input), level=self.level)
        return forecast.reset_index()


def log_statsforecast_model(
    fitted: FittedStatsForecast, name: str, level: int | None = None
) -> ModelInfo:
    """Log a fitted statsforecast model to the active MLflow run as a pyfunc model.

    level adds a level% prediction interval to every forecast. The fell_back flag is
    kept in the model metadata: a model that fell back is an AutoETS fit.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "statsforecast.pkl"
        fitted.sf.save(path)
        return mlflow.pyfunc.log_model(
            name=name,
            python_model=StatsForecastModel(),
            artifacts={ARTIFACT_KEY: str(path)},
            model_config={"level": [level]} if level else None,
            metadata={"fell_back": fitted.fell_back, "column": fitted.column},
        )
