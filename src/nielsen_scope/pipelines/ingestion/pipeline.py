from kedro.pipeline import Pipeline, node

from .nodes import (
    create_schema,
    fill_missing_weeks,
    ingest_isbn_static,
    ingest_sales_data,
    reload_tables,
    resample_weekly,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=create_schema,
                inputs=None,
                outputs="schema_ready",
                name="create_schema_node",
            ),
            #books data lane
            node(
                func=ingest_isbn_static,
                inputs="book_isbn_file",
                outputs="isbn_dim",
                name="ingest_isbn_static_node",
            ),
            #sales data lane
            node(
                func=ingest_sales_data,
                inputs="book_sales_file",
                outputs="sales_raw",
                name="ingest_sales_data_node",
            ),
            node(
                func=resample_weekly,
                inputs="sales_raw",
                outputs="sales_weekly",
                name="resample_weekly_node",
            ),
            node(
                func=fill_missing_weeks,
                inputs="sales_weekly",
                outputs="sales_filled",
                name="fill_missing_weeks_node",
            ),
            #one transaction: truncate, then categories -> books -> sales
            node(
                func=reload_tables,
                inputs=[
                    "isbn_dim",
                    "sales_filled",
                    "params:sheet_name_mappings",
                    "params:isbn_field_mappings",
                    "params:sales_field_mappings",
                    "schema_ready",
                ],
                outputs="tables_reloaded",
                name="reload_tables_node",
            ),
        ]
    )
    