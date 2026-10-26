from kedro.pipeline import Pipeline, node

from .nodes import cross_validate, select_book_features, train_final_xgb_model


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=select_book_features,
            inputs=["feature_table", "params:training"],
            outputs=["book_X", "book_y", "book_dates"],
            name="select_book_features_node",
        ),
        node(
            func=cross_validate,
            inputs=["book_X", "book_y", "params:training", "book_dates"],
            outputs=["training_metrics", "cv_predictions"],
            name="walk_forward_evaluate_xgb_node",
        ),
        node(
            func=train_final_xgb_model,
            inputs=["book_X", "book_y", "params:training"],
            outputs="book_xgb_model",
            name="train_final_xgb_model_node",
        )
    ])
