from pathlib import Path

import pandas as pd
import pytest
import yaml

from nielsen_scope.pipelines.ingestion.nodes import (
    _read_excel_concat,
    fill_missing_weeks,
    ingest_isbn_static,
    ingest_sales_data,
    resample_weekly,
)

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_SQL = ROOT / "src/nielsen_scope/sql/schema.sql"
PARAMETERS = yaml.safe_load((ROOT / "conf/base/parameters.yml").read_text())
ISBN_A = "9780000000001"
ISBN_B = "9780000000002"
DIM_COLUMNS = [
    "ISBN", "Title", "Author", "Imprint", "Publisher_Group", "RRP", "Binding",
    "Publication_Date", "Country_of_Publication", "Product_Class", "category",
]
BOOKS_COLUMNS = {
    "isbn", "title", "author", "imprint", "publisher_group", "rrp", "binding",
    "publication_date", "country_of_publication", "product_class", "category",
}
N_SCHEMA_STATEMENTS = 6


def _isbn_sheet(isbn: str, title: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ISBN": [isbn],
            "Title": [title],
            "Author": ["A. Author"],
            "Imprint": ["Imprint"],
            "Publisher Group": ["Group"],
            "RRP": [7.99],
            "Binding": ["Paperback"],
            "Publication Date": [pd.Timestamp("2010-01-01")],
            "Product Class": ["F1.1"],
            "Country of Publication": ["United Kingdom"],
            "Unused Column": ["x"],
        }
    )


def test_read_excel_concat_tags_category_and_snake_cases_columns():
    sheets = {
        "F - Adult Fiction": pd.DataFrame({"ISBN": [ISBN_A], " End Date ": ["2024-01-06"]}),
        "S Adult Non-Fiction Specialist": pd.DataFrame({"ISBN": [ISBN_B], "End Date": ["2024-01-06"]}),
    }
    out = _read_excel_concat(sheets)
    assert list(out["category"]) == ["F", "S"]
    assert "End_Date" in out.columns
    assert "sheet_name" not in out.columns


def test_read_excel_concat_without_isbn_raises():
    with pytest.raises(KeyError, match="No 'ISBN' column"):
        _read_excel_concat({"F - Adult Fiction": pd.DataFrame({"Title": ["x"]})})


def test_ingest_sales_data_parses_and_sorts_end_dates():
    sheets = {
        "F - Adult Fiction": pd.DataFrame(
            {"ISBN": [ISBN_A, ISBN_A], "End Date": ["2024-01-13", "2024-01-06"], "Volume": [1, 2]}
        )
    }
    out = ingest_sales_data(sheets)
    assert list(out["End_Date"]) == [pd.Timestamp("2024-01-06"), pd.Timestamp("2024-01-13")]
    assert list(out["Volume"]) == [2, 1]


def test_resample_weekly_sums_volume_and_value_and_averages_asp():
    daily = pd.DataFrame(
        {
            "ISBN": [ISBN_A, ISBN_A, ISBN_A],
            "End_Date": pd.to_datetime(["2024-01-06", "2024-01-04", "2024-01-13"]),
            "Volume": [3, 2, 1],
            "Value": [30.0, 24.0, 9.0],
            "ASP": [10.0, 12.0, 9.0],
            "category": ["F", "F", "F"],
        }
    )
    out = resample_weekly(daily)
    assert list(out["End_Date"]) == [pd.Timestamp("2024-01-07"), pd.Timestamp("2024-01-14")]
    assert list(out["Volume"]) == [5, 1]
    assert list(out["Value"]) == [54.0, 9.0]
    assert list(out["ASP"]) == [11.0, 9.0]
    assert list(out["category"]) == ["F", "F"]


def test_fill_missing_weeks_zero_fills_gaps_up_to_the_last_week():
    weekly = pd.DataFrame(
        {
            "ISBN": [ISBN_A, ISBN_A, ISBN_B],
            "End_Date": pd.to_datetime(["2024-01-07", "2024-01-21", "2024-01-14"]),
            "Volume": [5, 7, 4],
            "Value": [50.0, 70.0, 40.0],
            "ASP": [10.0, 10.0, 10.0],
            "category": ["F", "F", "S"],
        }
    )
    out = fill_missing_weeks(weekly)
    assert len(out) == 5
    a = out[out["ISBN"] == ISBN_A].set_index("End_Date")
    assert a.loc[pd.Timestamp("2024-01-14"), "Volume"] == 0
    assert a.loc[pd.Timestamp("2024-01-14"), "category"] == "F"
    b = out[out["ISBN"] == ISBN_B].sort_values("End_Date")
    assert list(b["End_Date"]) == [pd.Timestamp("2024-01-14"), pd.Timestamp("2024-01-21")]
    assert list(b["Volume"]) == [4, 0]
    assert set(b["category"]) == {"S"}
    assert out[["Volume", "Value", "ASP"]].notna().all().all()


def test_ingest_isbn_static_keeps_country_of_publication_and_last_duplicate():
    sheets = {
        "F - Adult Fiction": _isbn_sheet(ISBN_A, "First Title"),
        "S Adult Non-Fiction Specialist": _isbn_sheet(ISBN_A, "Second Title"),
    }
    out = ingest_isbn_static(sheets)
    assert list(out.columns) == DIM_COLUMNS
    assert len(out) == 1
    row = out.iloc[0]
    assert row["Title"] == "Second Title"
    assert row["category"] == "S"
    assert row["Country_of_Publication"] == "United Kingdom"


def test_isbn_mapping_renames_every_dimension_column_to_a_books_column():
    out = ingest_isbn_static({"F - Adult Fiction": _isbn_sheet(ISBN_A, "Title")})
    renamed = out.rename(columns=PARAMETERS["isbn_field_mappings"])
    assert set(renamed.columns) <= BOOKS_COLUMNS
    assert "country_of_publication" in renamed.columns


def test_schema_sql_splits_into_create_statements_only():
    statements = [s.strip() for s in SCHEMA_SQL.read_text().split(";") if s.strip()]
    assert len(statements) == N_SCHEMA_STATEMENTS
    assert all(s.startswith("CREATE ") for s in statements)