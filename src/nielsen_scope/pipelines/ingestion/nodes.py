from pathlib import Path
from sqlalchemy import create_engine, text
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def create_schema(connection_string: str) -> None:
    """
    Create the PostgreSQL schema if it does not already exist.
    """
    #Locate schema.sql
    schema_path = (
        Path(__file__).parents[2]
        / "sql"
        / "schema.sql"
    )

    if not schema_path.exists():
        raise FileNotFoundError(
            f"Schema file not found: {schema_path}"
        )

    #Read SQL file
    with open(schema_path, "r") as f:
        sql = f.read()

    #Connect to PostgreSQL
    print(f"Connecting to PostgreSQL: {connection_string}")
    engine = create_engine(connection_string)

    #Execute each SQL statement
    with engine.begin() as conn:
        for statement in sql.split(";"):
            statement = statement.strip()
            if statement:
                conn.execute(text(statement))

    print("Schema created successfully.")
    return "schema_ready"

def _convert_fieldnames(df = pd.DataFrame) -> pd.DataFrame:
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
        print(f"Excel file read successfully.")

        return df

    except Exception as e:
        print(f"Error loading excel dataset: {e}")
        raise

def ingest_sales_data(sheets: dict) -> pd.DataFrame:
    """Read raw sales data excel, ingest -> long, sorted, fact-grain frame."""
    df = _read_excel_concat(sheets)

    #replace whitespace with _ within fieldnames
    df = _convert_fieldnames(df)

    if "End_Date" not in df.columns:
        raise KeyError(f"Expected sales data with 'End_Date' in Sales Dataset")

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
        "ISBN", "Title", "Author", "Imprint", "Publisher_Group",
        "RRP", "Binding", "Publication_Date", "Product_Class", "category"
    ]
    df = df[dim_cols].drop_duplicates(subset="ISBN", keep="last")
    return df

def load_sales(df_fact_sales: pd.DataFrame, connection_string: str,
                sales_mapping_dictionary: dict, _books_loaded: str) -> None:
    """Load the sales fact data into sql table."""
    df = df_fact_sales.rename(columns=sales_mapping_dictionary)
    df = df.drop(columns=["sheet_name", "category"], errors="ignore")
    engine = create_engine(connection_string)
    df.to_sql("sales", engine, if_exists="append", index=False)
    return "sales_loaded"

def load_books(df_isbn: pd.DataFrame, connection_string: str,
                isbn_mapping_dictionary: dict, _categories_loaded: str) -> str:
    """Load the isbn static/dimension data into sql table."""
    df = df_isbn.rename(columns=isbn_mapping_dictionary)
    engine = create_engine(connection_string)
    df.to_sql("books", engine, if_exists="append", index=False)
    return "books_loaded"

def load_categories(sheet_name_mappings: dict, connection_string: str, 
                    _tables_truncated: str) -> str:
    df = pd.DataFrame(
        [{"category": k, "category_name": v} for k, v in sheet_name_mappings.items()]
    )
    engine = create_engine(connection_string)
    df.to_sql("categories", engine, if_exists="append", index=False)
    return "categories_loaded"

def truncate_tables(connection_string: str, _schema_ready: str) -> str:
    """Empty the tables, keeping constraints, so a full load is idempotent."""
    engine = create_engine(connection_string)
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE sales, books, categories RESTART IDENTITY CASCADE"))
    return "tables_truncated"




    