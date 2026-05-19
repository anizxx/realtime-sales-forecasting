"""Data ingestion for the real-time sales forecasting pipeline.

This module loads sales data from a local CSV or HTTP(S) CSV endpoint, validates
the expected Rossmann-style schema, normalizes data types, and writes the clean
raw records into PostgreSQL.
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import SQLAlchemyError


LOGGER = logging.getLogger("sales_forecasting.ingestion")

REQUIRED_COLUMNS = {
    "Date",
    "Store",
    "Sales",
    "Customers",
    "Open",
    "Promo",
    "StateHoliday",
    "SchoolHoliday",
}

INTEGER_COLUMNS = ["Store", "Sales", "Customers", "Open", "Promo", "SchoolHoliday"]
DEFAULT_TABLE_NAME = "raw_sales"


@dataclass(frozen=True)
class IngestionConfig:
    """Runtime settings for the ingestion job."""

    source: str
    postgres_url: str | None
    table_name: str = DEFAULT_TABLE_NAME
    if_exists: str = "append"
    chunksize: int = 10_000
    dry_run: bool = False


class SchemaValidationError(ValueError):
    """Raised when input data does not match the required sales schema."""


def configure_logging(level: str = "INFO") -> None:
    """Configure process-wide structured logging for ingestion runs."""

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_sales_data(source: str) -> pd.DataFrame:
    """Load sales data from a local CSV path or an HTTP(S) CSV URL.

    Args:
        source: Local file path or HTTP(S) URL that returns CSV data.

    Returns:
        Loaded sales records as a pandas DataFrame.

    Raises:
        FileNotFoundError: If a local CSV path does not exist.
        ValueError: If the source cannot be read as CSV.
    """

    LOGGER.info("Loading sales data from %s", source)
    is_url = source.startswith(("http://", "https://"))

    if not is_url and not Path(source).exists():
        raise FileNotFoundError(f"Input CSV not found: {source}")

    try:
        dataframe = pd.read_csv(source)
    except Exception as exc:  # pandas raises parser, URL, and decoding errors.
        raise ValueError(f"Failed to read CSV source '{source}': {exc}") from exc

    LOGGER.info("Loaded %d rows and %d columns", len(dataframe), len(dataframe.columns))
    return dataframe


def validate_schema(dataframe: pd.DataFrame, required_columns: Iterable[str] = REQUIRED_COLUMNS) -> None:
    """Validate that the DataFrame contains all required input columns.

    Args:
        dataframe: Input sales records.
        required_columns: Required column names.

    Raises:
        SchemaValidationError: If required columns are missing or data is empty.
    """

    if dataframe.empty:
        raise SchemaValidationError("Input data is empty.")

    missing_columns = sorted(set(required_columns) - set(dataframe.columns))
    if missing_columns:
        raise SchemaValidationError(f"Missing required columns: {missing_columns}")


def normalize_sales_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Normalize sales records for storage and downstream feature engineering.

    Args:
        dataframe: Raw sales records with required columns.

    Returns:
        A normalized DataFrame with clean dates, numeric fields, and duplicate
        Store-Date rows removed.

    Raises:
        SchemaValidationError: If type coercion fails or required values are null.
    """

    validate_schema(dataframe)
    normalized = dataframe.copy()

    try:
        normalized["Date"] = pd.to_datetime(normalized["Date"], errors="raise").dt.date
        for column in INTEGER_COLUMNS:
            normalized[column] = pd.to_numeric(normalized[column], errors="raise").astype("int64")
    except Exception as exc:
        raise SchemaValidationError(f"Failed to normalize column types: {exc}") from exc

    normalized["StateHoliday"] = normalized["StateHoliday"].fillna("0").astype(str).str.strip()

    null_counts = normalized[list(REQUIRED_COLUMNS)].isna().sum()
    columns_with_nulls = null_counts[null_counts > 0].to_dict()
    if columns_with_nulls:
        raise SchemaValidationError(f"Required columns contain nulls: {columns_with_nulls}")

    before = len(normalized)
    normalized = normalized.drop_duplicates(subset=["Store", "Date"], keep="last")
    dropped = before - len(normalized)
    if dropped:
        LOGGER.warning("Dropped %d duplicate Store-Date rows", dropped)

    normalized = normalized.sort_values(["Store", "Date"]).reset_index(drop=True)
    return normalized


def build_postgres_engine(postgres_url: str) -> Engine:
    """Create a SQLAlchemy engine for PostgreSQL.

    Args:
        postgres_url: SQLAlchemy-compatible PostgreSQL connection URL.

    Returns:
        Configured SQLAlchemy engine.
    """

    return create_engine(postgres_url, pool_pre_ping=True, future=True)


def ensure_raw_sales_indexes(engine: Engine, table_name: str) -> None:
    """Create helpful indexes for the raw sales table when using PostgreSQL.

    Args:
        engine: SQLAlchemy database engine.
        table_name: Destination table name.
    """

    safe_table_name = "".join(character for character in table_name if character.isalnum() or character == "_")
    if safe_table_name != table_name:
        raise ValueError("Table name may only contain letters, numbers, and underscores.")

    index_sql = f"""
    CREATE INDEX IF NOT EXISTS idx_{safe_table_name}_store_date
    ON {safe_table_name} ("Store", "Date");
    """

    with engine.begin() as connection:
        connection.execute(text(index_sql))


def write_to_postgres(
    dataframe: pd.DataFrame,
    engine: Engine,
    table_name: str = DEFAULT_TABLE_NAME,
    if_exists: str = "append",
    chunksize: int = 10_000,
) -> int:
    """Write normalized sales data to PostgreSQL.

    Args:
        dataframe: Normalized sales records.
        engine: SQLAlchemy database engine.
        table_name: Destination table.
        if_exists: pandas to_sql behavior: append, replace, or fail.
        chunksize: Number of rows per insert batch.

    Returns:
        Number of records written.

    Raises:
        SQLAlchemyError: If the database write fails.
    """

    LOGGER.info("Writing %d rows to PostgreSQL table '%s'", len(dataframe), table_name)
    try:
        dataframe.to_sql(
            table_name,
            engine,
            if_exists=if_exists,
            index=False,
            chunksize=chunksize,
            method="multi",
        )
        ensure_raw_sales_indexes(engine, table_name)
    except SQLAlchemyError:
        LOGGER.exception("Database write failed")
        raise

    LOGGER.info("Finished writing %d rows", len(dataframe))
    return len(dataframe)


def run_ingestion(config: IngestionConfig) -> pd.DataFrame:
    """Run the full ingestion workflow.

    Args:
        config: Ingestion job settings.

    Returns:
        The normalized DataFrame, which is useful for tests and dry runs.

    Raises:
        ValueError: If PostgreSQL is required but not configured.
    """

    raw_data = load_sales_data(config.source)
    normalized_data = normalize_sales_data(raw_data)

    if config.dry_run:
        LOGGER.info("Dry run complete. Validated %d normalized rows.", len(normalized_data))
        return normalized_data

    if not config.postgres_url:
        raise ValueError("POSTGRES_URL or --postgres-url is required unless --dry-run is set.")

    engine = build_postgres_engine(config.postgres_url)
    write_to_postgres(
        normalized_data,
        engine=engine,
        table_name=config.table_name,
        if_exists=config.if_exists,
        chunksize=config.chunksize,
    )
    return normalized_data


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the ingestion job."""

    parser = argparse.ArgumentParser(description="Ingest Rossmann-style sales data into PostgreSQL.")
    parser.add_argument("--source", required=True, help="CSV file path or HTTP(S) CSV URL.")
    parser.add_argument(
        "--postgres-url",
        default=os.getenv("POSTGRES_URL"),
        help="PostgreSQL SQLAlchemy URL. Defaults to POSTGRES_URL environment variable.",
    )
    parser.add_argument("--table-name", default=DEFAULT_TABLE_NAME, help="Destination table name.")
    parser.add_argument(
        "--if-exists",
        choices=["append", "replace", "fail"],
        default="append",
        help="Behavior when the destination table already exists.",
    )
    parser.add_argument("--chunksize", type=int, default=10_000, help="Rows per database insert batch.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and normalize without writing to database.")
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    return parser.parse_args()


def main() -> None:
    """CLI entry point for data ingestion."""

    args = parse_args()
    configure_logging(args.log_level)

    config = IngestionConfig(
        source=args.source,
        postgres_url=args.postgres_url,
        table_name=args.table_name,
        if_exists=args.if_exists,
        chunksize=args.chunksize,
        dry_run=args.dry_run,
    )
    run_ingestion(config)


if __name__ == "__main__":
    main()
