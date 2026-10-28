from nielsen_scope.pipeline_registry import DESTRUCTIVE_PIPELINES, default_pipeline
from nielsen_scope.pipelines.feature_engineering.pipeline import (
    create_pipeline as feature_engineering_pipeline,
)
from nielsen_scope.pipelines.ingestion.pipeline import (
    create_pipeline as ingestion_pipeline,
)
from nielsen_scope.pipelines.training.pipeline import (
    create_pipeline as training_pipeline,
)

TRAINING_PERSIST_NODES = [
    "select_book_features_node",
    "walk_forward_evaluate_node",
    "write_forecasts_node",
    "write_model_metrics_node",
]


def _pipelines() -> dict:
    return {
        "ingestion": ingestion_pipeline(),
        "feature_engineering": feature_engineering_pipeline(),
        "training": training_pipeline(),
    }


def _names(pipeline) -> list[str]:
    return [node.name for node in pipeline.nodes]


def test_ingestion_is_the_destructive_pipeline_and_really_truncates():
    assert "ingestion" in DESTRUCTIVE_PIPELINES
    assert "truncate_tables_node" in _names(_pipelines()["ingestion"])


def test_default_pipeline_leaves_out_ingestion():
    names = _names(default_pipeline(_pipelines()))
    assert "truncate_tables_node" not in names
    assert not any("truncate" in name or "drop" in name for name in names)
    assert set(_names(_pipelines()["training"])) <= set(names)
    assert set(_names(_pipelines()["feature_engineering"])) <= set(names)


def test_persisting_the_evaluation_runs_four_nodes_and_no_registration():
    names = _names(training_pipeline().to_nodes("write_model_metrics_node"))
    assert sorted(names) == sorted(TRAINING_PERSIST_NODES)
