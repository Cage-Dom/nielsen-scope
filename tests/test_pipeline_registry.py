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
from nielsen_scope.pipelines.training.pipeline import instance_pipeline, load_grid

INSTANCE_NODES = [
    "bind_run_params_node",
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


def test_instance_pipeline_is_namespaced_and_registers_no_model():
    pipe = instance_pipeline("9780722532935", "xgboost")
    namespace = "9780722532935.xgboost"
    assert sorted(_names(pipe)) == sorted(f"{namespace}.{name}" for name in INSTANCE_NODES)
    assert not any("book_xgb_model" in output for output in pipe.outputs())
    assert not any("train_final" in name for name in _names(pipe))


def test_instances_share_the_feature_table_and_the_shared_params():
    pipe = instance_pipeline("9780722532935", "naive")
    assert pipe.inputs() == {"feature_table", "params:training", "params:postgres_connection"}


def test_training_pipeline_has_one_instance_per_isbn_and_model():
    grid = load_grid()
    expected = len(grid["isbns"]) * len(grid["models"]) * len(INSTANCE_NODES)
    assert len(training_pipeline().nodes) == expected
    one = training_pipeline().only_nodes_with_namespaces(
        [f"{grid['isbns'][0]}.{grid['models'][0]}"]
    )
    assert len(one.nodes) == len(INSTANCE_NODES)
