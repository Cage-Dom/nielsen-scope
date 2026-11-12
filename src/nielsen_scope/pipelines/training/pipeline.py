import os
from functools import partial
from pathlib import Path

from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings
from kedro.pipeline import Pipeline, node, pipeline
from kedro.utils import find_kedro_project

from .nodes import cross_validate, select_book_features
from .persist import write_results

def load_grid() -> dict:
    """The (ISBN, model) grid from `training_grid` in parameters, read when the pipeline
    is built, with the run's config environments: conf/base, then KEDRO_ENV or the
    default run environment (conf/local).

    `kedro run --env X` is not visible here (Kedro hands it to the session only), so
    use KEDRO_ENV=X. `--params` and `--conf-source` are not applied here either.
    """
    project_path = find_kedro_project(Path.cwd())
    if project_path is None:
        raise RuntimeError(
            "load_grid: no Kedro project (pyproject.toml with [tool.kedro]) at or above "
            f"{Path.cwd()}"
        )
    loader = OmegaConfigLoader(
        conf_source=str(project_path / settings.CONF_SOURCE),
        env=os.getenv("KEDRO_ENV"),
        **settings.CONFIG_LOADER_ARGS,
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
            func=write_results,
            inputs=["cv_predictions", "training_metrics", "cv_fold_metrics", "run_params"],
            outputs="results_written",
            name="write_results_node",
        ),
    ])
    return pipeline(
        base,
        namespace=f"{isbn}.{model_type}",
        inputs={"feature_table"},
        parameters={"training"},
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
