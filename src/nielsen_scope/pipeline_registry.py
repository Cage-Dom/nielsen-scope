"""Project pipelines."""
from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

# Pipelines that delete data. A bare `kedro run` must never include them, so they
# run only when named: `kedro run --pipelines=ingestion`.
DESTRUCTIVE_PIPELINES = {"ingestion"}


def default_pipeline(pipelines: dict[str, Pipeline]) -> Pipeline:
    """Every pipeline except the destructive ones."""
    return sum(
        (pipe for name, pipe in pipelines.items() if name not in DESTRUCTIVE_PIPELINES),
        start=Pipeline([]),
    )


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = find_pipelines(raise_errors=True)
    pipelines["__default__"] = default_pipeline(pipelines)
    return pipelines
