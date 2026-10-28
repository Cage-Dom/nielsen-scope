from functools import partial
from pathlib import Path

from kedro.config import OmegaConfigLoader
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import cross_validate, select_book_features
from .persist import write_forecasts, write_model_metrics

CONF_PATH = Path(__file__).resolve().parents[4] / "conf"


def load_grid() -> dict:
    """The (ISBN, model) grid from conf/base/parameters.yml `training_grid`, read when the
    pipeline is built. Reads the base environment only, so it needs no conf/local."""
    loader = OmegaConfigLoader(
        conf_source=str(CONF_PATH), base_env="base", default_run_env="base"
    )
    return loader["parameters"]["training_grid"]


def bind_run_params(training: dict, isbn: str, model_type: str) -> dict:
    """The shared training params with this instance's ISBN and model type set."""
    return {**training, "isbn": isbn, "model_type": model_type}


def instance_pipeline(isbn: str, model_type: str) -> Pipeline:
    """Evaluate one model on one book and persist the results, namespaced `<isbn>.<model_type>`."""
    base = Pipeline([
        node(
            func=partial(bind_run_params, isbn=isbn, model_type=model_type),
            inputs="params:training",
            outputs="run_params",
            name="bind_run_params_node",
        ),
        node(
            func=select_book_features,
            inputs=["feature_table", "run_params"],
            outputs=["book_X", "book_y", "book_dates"],
            name="select_book_features_node",
        ),
        node(
            func=cross_validate,
            inputs=["book_X", "book_y", "run_params", "book_dates"],
            outputs=["training_metrics", "cv_predictions", "cv_fold_metrics"],
            name="walk_forward_evaluate_node",
        ),
        node(
            func=write_forecasts,
            inputs=["cv_predictions", "run_params", "params:postgres_connection"],
            outputs="forecasts_written",
            name="write_forecasts_node",
        ),
        node(
            func=write_model_metrics,
            inputs=[
                "training_metrics",
                "cv_fold_metrics",
                "run_params",
                "params:postgres_connection",
                "forecasts_written",
            ],
            outputs="model_metrics_written",
            name="write_model_metrics_node",
        ),
    ])
    return pipeline(
        base,
        namespace=f"{isbn}.{model_type}",
        inputs={"feature_table"},
        parameters={"training", "postgres_connection"},
    )


def create_pipeline(**kwargs) -> Pipeline:
    grid = load_grid()
    return sum(
        (
            instance_pipeline(str(isbn), model_type)
            for isbn in grid["isbns"]
            for model_type in grid["models"]
        ),
        start=Pipeline([]),
    )
