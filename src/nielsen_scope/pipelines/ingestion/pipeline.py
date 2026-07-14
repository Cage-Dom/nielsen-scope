from kedro.pipeline import Pipeline, node

from .nodes import create_schema, truncate_tables, load_categories, ingest_isbn_static, load_books, ingest_sales_data, load_sales, resample_weekly, fill_missing_weeks

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=create_schema,
                inputs="params:postgres_connection",
                outputs="schema_ready",
                name="create_schema_node",
            ),
            node(
                func=truncate_tables,
                inputs=["params:postgres_connection", "schema_ready"],
                outputs="tables_truncated",
                name="truncate_tables_node"
            ),
            node(
                func=load_categories,
                inputs=["params:sheet_name_mappings", "params:postgres_connection", "tables_truncated"],
                outputs="categories_loaded",
                name="load_categories_node"
            ),
            #books data lane
            node(
                func=ingest_isbn_static,
                inputs="book_isbn_file",
                outputs="isbn_dim",
                name="ingest_isbn_static_node"
            ),
            node(
                func=load_books,
                inputs=["isbn_dim", "params:postgres_connection", "params:isbn_field_mappings", "categories_loaded"],
                outputs="books_loaded",
                name="load_books_node"
            ),
            #sales data lane
            node(
                func=ingest_sales_data,
                inputs="book_sales_file",
                outputs="sales_raw",
                name="ingest_sales_data_node"
            ),
            node(
                func=resample_weekly,
                inputs="sales_raw",
                outputs="sales_weekly",
                name="resample_weekly_node"
            ),
            node(
                func=fill_missing_weeks,
                inputs="sales_weekly",
                outputs="sales_filled",
                name="fill_missing_weeks_node"
            ),
            node(
                func=load_sales,
                inputs=["sales_filled", "params:postgres_connection", "params:sales_field_mappings", "books_loaded"],
                outputs="sales_loaded",
                name="load_sales_node"
            )
        ]
    )
