from kedro.pipeline import Pipeline, node 
from .nodes import select_book_features, walk_forward_evaluate_xgb, train_final_xgb_model


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=select_book_features,
            inputs=["feature_table", "params:training"],
            outputs=["book_X", "book_y"],
            name="select_book_features_node",
        ),
        node(
            func=walk_forward_evaluate_xgb,
            inputs=["book_X", "book_y", "params:training"],
            outputs="training_metrics",
            name="walk_forward_evaluate_xgb_node",
        ),
        node(
            func=train_final_xgb_model,
            inputs=["book_X", "book_y", "params:training"],
            outputs="book_xgb_model",
            name="train_final_xgb_model_node",
        )
    ])
