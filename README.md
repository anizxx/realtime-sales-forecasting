# Real-Time Sales Forecasting Pipeline

Production-grade forecasting pipeline for Rossmann-style store sales data.

## Phase 1: Data Ingestion

The ingestion job loads a CSV from disk or an HTTP(S) URL, validates the required
schema, normalizes dates and integer fields, removes duplicate `Store`/`Date`
rows, and writes the cleaned raw records to PostgreSQL.

Required input columns:

```text
Date, Store, Sales, Customers, Open, Promo, StateHoliday, SchoolHoliday
```

Install Phase 1 dependencies:

```bash
pip install -r requirements.txt
```

Run the local pipeline with the included sample data:

```bash
python run_pipeline.py
```

Run the local pipeline with your own CSV:

```bash
python run_pipeline.py --source path/to/train.csv --output data/engineered_sales.csv
```

The full ML, API, dashboard, and Airflow stack is pinned in
`requirements-full-py311.txt` and should be installed from a Python 3.11
virtual environment. Airflow 2.10.3 is not compatible with Python 3.13.

Validate a CSV without writing to PostgreSQL:

```bash
python data_ingestion.py --source data/train.csv --dry-run
```

Write to PostgreSQL:

```bash
set POSTGRES_URL=postgresql+psycopg2://sales_user:sales_password@localhost:5432/sales_forecasting
python data_ingestion.py --source data/train.csv --table-name raw_sales --if-exists append
```

You can also copy `.env.example` into your local environment manager and set
`POSTGRES_URL` there.

## Planned Phases

1. Data ingestion
2. Feature engineering
3. EDA notebook
4. Model training
5. Model evaluation
6. Airflow orchestration
7. Streamlit dashboard
8. FastAPI endpoint
9. Docker and production hardening
