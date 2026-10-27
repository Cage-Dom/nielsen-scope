from kedro.pipeline import Pipeline, node

from .nodes import cross_validate, select_book_features, train_final_xgb_model
from .persist import write_forecasts, write_model_metrics


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
            outputs=["training_metrics", "cv_predictions", "cv_fold_metrics"],
            name="walk_forward_evaluate_xgb_node",
        ),
        node(
            func=write_forecasts,
            inputs=["cv_predictions", "params:training", "params:postgres_connection"],
            outputs="forecasts_written",
            name="write_forecasts_node",
        ),
        node(
            func=write_model_metrics,
            inputs=[
                "training_metrics",
                "cv_fold_metrics",
                "params:training",
                "params:postgres_connection",
                "forecasts_written",
            ],
            outputs="model_metrics_written",
            name="write_model_metrics_node",
        ),
        node(
            func=train_final_xgb_model,
            inputs=["book_X", "book_y", "params:training"],
            outputs="book_xgb_model",
            name="train_final_xgb_model_node",
        )
    ])
