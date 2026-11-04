import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection

from nielsen_scope.db import db_url

logger = logging.getLogger(__name__)


def create_schema() -> str:
    """Create the PostgreSQL schema if it does not already exist."""
    schema_path = Path(__file__).parents[2] / "sql" / "schema.sql"
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")
    sql = schema_path.read_text()

    engine = create_engine(db_url())
    try:
        with engine.begin() as conn:
            # schema.sql holds CREATE statements only, with no ';' in comments or
            # literals (pinned by test_schema_sql_splits_into_create_statements_only)
            for statement in sql.split(";"):
                statement = statement.strip()
                if statement:
                    conn.execute(text(statement))
    finally:
        engine.dispose()
    logger.info("create_schema: applied %s", schema_path.name)
    return "schema_ready"


def _convert_fieldnames(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all spaces in column names from input excel into snake case"""
    df.columns = df.columns.str.strip().str.replace(' ', '_')
    return df


def _read_excel_concat(sheets: dict) -> pd.DataFrame:
    """
    Helper function shared across both datasets to concat all sheets.
    Tags each row with a single-letter 'category' code derived from the
    sheet name (e.g. 'F - Adult Fiction' -> 'F').
    """
    try:
       #concatenate all sheets
        df = pd.concat(
            [
                #replace whitespace with _ within fieldnames
                _convert_fieldnames(sheet_df).assign(sheet_name=sheet_name)
                for sheet_name, sheet_df in sheets.items()
            ],
            ignore_index=True
        )
        if "ISBN" not in df.columns:
            raise KeyError(
                f"No 'ISBN' column after concat; sheets={list(sheets)}, "
                f"columns={list(df.columns)}")
        
        #rename for mapping later and trim so just the letter identifier
        df = df.rename(columns={'sheet_name': 'category'})
        df['category'] = df['category'].apply(lambda x: x[0])
        logger.info("Excel file read successfully.")

        return df

    except Exception as e:
        logger.error(f"Error loading excel dataset: {e}")
        raise

def ingest_sales_data(sheets: dict) -> pd.DataFrame:
    """Read raw sales data excel, ingest -> long, sorted, fact-grain frame."""
    df = _read_excel_concat(sheets)

    #replace whitespace with _ within fieldnames
    df = _convert_fieldnames(df)

    if "End_Date" not in df.columns:
        raise KeyError("Expected sales data with 'End_Date' in Sales Dataset")

    df["End_Date"] = pd.to_datetime(df["End_Date"])
    return df.sort_values("End_Date").reset_index(drop=True)

    
def resample_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sales data to weekly per ISBN"""
    df = (
        df
        .groupby("ISBN")
        .resample("1W", on="End_Date")
        .agg({
            "Volume": "sum",
            "Value": "sum",
            "ASP": "mean",
            "category": "first",
        })
        .reset_index()
    )
    return df


def fill_missing_weeks(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure every ISBN has a continuous weekly series, zero-filled."""
    bounds = df.groupby("ISBN")["End_Date"].min().reset_index()
    bounds.columns = ["ISBN", "Min_Date"]
    bounds["Max_Date"] = df["End_Date"].max()

    #capture category per ISBN
    isbn_category = df.drop_duplicates("ISBN").set_index("ISBN")["category"]

    #capture min max dates per isbn
    grids = [
        pd.DataFrame({
            "End_Date": pd.date_range(row.Min_Date, row.Max_Date, freq="1W"),
            "ISBN": row.ISBN,
        })
        for row in bounds.itertuples(index=False)
    ]
    complete_grid = pd.concat(grids, ignore_index=True)

    #join grid
    df = complete_grid.merge(df, on=["End_Date", "ISBN"], how="left")
    #fill null with 0
    df[["Volume", "Value", "ASP"]] = df[["Volume", "Value", "ASP"]].fillna(0)
    #fill null from captured lookup
    df["category"] = df["category"].fillna(df["ISBN"].map(isbn_category))
    
    return df

def ingest_isbn_static(sheets: dict) -> pd.DataFrame:
    """Read static ISBN attribute excel -> dimension frame."""
    df = _read_excel_concat(sheets)
    #normalise names and remove whitespace
    df = _convert_fieldnames(df)

    dim_cols = [
        "ISBN", "Title", "Author", "Imprint", "Publisher_Group", "RRP", "Binding",
        "Publication_Date", "Country_of_Publication", "Product_Class", "category"
    ]
    df = df[dim_cols].drop_duplicates(subset="ISBN", keep="last")
    return df

def truncate_tables(conn: Connection) -> None:
    """Empty sales, books and categories inside the caller's transaction."""
    conn.execute(text("TRUNCATE TABLE sales, books, categories RESTART IDENTITY CASCADE"))


def load_categories(conn: Connection, sheet_name_mappings: dict) -> None:
    df = pd.DataFrame(
        [{"category": k, "category_name": v} for k, v in sheet_name_mappings.items()]
    )
    df.to_sql("categories", conn, if_exists="append", index=False)


def load_books(conn: Connection, df_isbn: pd.DataFrame, isbn_mapping_dictionary: dict) -> None:
    """Load the isbn static/dimension data into the books table."""
    df = df_isbn.rename(columns=isbn_mapping_dictionary)
    df.to_sql("books", conn, if_exists="append", index=False)


def load_sales(
    conn: Connection, df_fact_sales: pd.DataFrame, sales_mapping_dictionary: dict
) -> None:
    """Load the sales fact data into the sales table."""
    df = df_fact_sales.rename(columns=sales_mapping_dictionary)
    df = df.drop(columns=["sheet_name", "category"], errors="ignore")
    df.to_sql("sales", conn, if_exists="append", index=False)


def reload_tables(
    isbn_dim: pd.DataFrame,
    sales_filled: pd.DataFrame,
    sheet_name_mappings: dict,
    isbn_field_mappings: dict,
    sales_field_mappings: dict,
    _schema_ready: str,
) -> str:
    """Truncate and reload categories, books and sales in one transaction.

    TRUNCATE is transactional in PostgreSQL, so a failed load rolls back to the
    previous contents instead of leaving sales empty and books full.
    """
    engine = create_engine(db_url())
    try:
        with engine.begin() as conn:
            truncate_tables(conn)
            load_categories(conn, sheet_name_mappings)
            load_books(conn, isbn_dim, isbn_field_mappings)
            load_sales(conn, sales_filled, sales_field_mappings)
    finally:
        engine.dispose()
    logger.info("reload_tables: %d books, %d sales rows", len(isbn_dim), len(sales_filled))
    return "tables_reloaded"




    