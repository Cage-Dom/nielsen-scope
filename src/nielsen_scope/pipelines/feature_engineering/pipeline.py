from kedro.pipeline import Pipeline, node

from .nodes import build_feature_table


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=build_feature_table,
                inputs="sales_features",
                outputs="feature_table",
                name="build_feature_table_node",
            ),
        ]
    )