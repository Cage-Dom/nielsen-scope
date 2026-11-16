from kedro.pipeline import Pipeline, node

from .nodes import build_feature_table
from .persist import write_feature_snapshot


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=build_feature_table,
                inputs="sales_features",
                outputs="feature_table",
                name="build_feature_table_node",
            ),
            node(
                func=write_feature_snapshot,
                inputs="feature_table",
                outputs=None,
                name="write_feature_snapshot_node"
            ),
        ]
    )
