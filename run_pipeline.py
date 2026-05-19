"""Local runner for the sales forecasting pipeline phases currently available.

The production pipeline will eventually be orchestrated by Airflow. This small
runner exists so the project can be executed from one beginner-friendly terminal
command while we build the phases incrementally.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from data_ingestion import IngestionConfig, configure_logging as configure_ingestion_logging
from data_ingestion import run_ingestion
from feature_engineering import process_features


LOGGER = logging.getLogger("sales_forecasting.runner")


def run_local_pipeline(source: str, output: str) -> Path:
    """Validate raw sales data and write engineered features to disk.

    Args:
        source: Input CSV path.
        output: Destination CSV path for engineered features.

    Returns:
        Path to the engineered feature CSV.
    """

    configure_ingestion_logging("INFO")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Starting local sales forecasting pipeline")
    normalized_data = run_ingestion(
        IngestionConfig(
            source=source,
            postgres_url=None,
            dry_run=True,
        )
    )

    engineered_data = process_features(normalized_data)
    engineered_data.to_csv(output_path, index=False)
    LOGGER.info("Pipeline finished. Engineered features saved to %s", output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the local pipeline runner."""

    parser = argparse.ArgumentParser(description="Run the available local sales forecasting pipeline phases.")
    parser.add_argument("--source", default="data/sample_sales.csv", help="Input sales CSV path.")
    parser.add_argument("--output", default="data/engineered_sales.csv", help="Output engineered CSV path.")
    return parser.parse_args()


def main() -> None:
    """CLI entry point for the local pipeline runner."""

    args = parse_args()
    run_local_pipeline(source=args.source, output=args.output)


if __name__ == "__main__":
    main()
