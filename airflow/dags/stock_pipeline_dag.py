"""
Task 5.3 — orchestrates the full daily refresh: ingest -> transform -> retrain.

    ingest_market_data  ->  transform_with_dbt  ->  retrain_models

- ingest_market_data stands in for the Airbyte connector described in the
  README: it lands fresh OHLCV bars into the `raw_prices` table of a local
  DuckDB warehouse. (Running the actual Airbyte platform — a multi-container
  ELT tool — isn't practical for a local portfolio demo; this task is the
  honest, runnable equivalent of what that connector would deliver.)
- transform_with_dbt runs the dbt project in ../dbt_stock (staging -> marts:
  daily returns, rolling SMA/volatility, latest-per-ticker snapshot).
- retrain_models shells out to deployment/train.py in the separate
  deployment_venv (TensorFlow lives there, not in Airflow's own venv), so a
  full run keeps the FastAPI service's models fresh.

Test locally with:
    airflow dags test stock_price_pipeline 2026-01-01
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import duckdb
import pandas as pd
import pendulum
import yfinance as yf
from airflow.decorators import dag, task
from airflow.exceptions import AirflowException

AIRFLOW_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = AIRFLOW_DIR.parent
WAREHOUSE_DB = AIRFLOW_DIR / "warehouse" / "stock.duckdb"
DBT_PROJECT_DIR = AIRFLOW_DIR / "dbt_stock"
DEPLOYMENT_VENV_PYTHON = REPO_ROOT / "deployment_venv" / "Scripts" / "python.exe"
TRAIN_SCRIPT = REPO_ROOT / "deployment" / "train.py"
DBT_BIN = Path(sys.executable).with_name("dbt.exe" if os.name == "nt" else "dbt")

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]


@dag(
    dag_id="stock_price_pipeline",
    description="Ingest OHLCV data, transform with dbt, retrain forecasting models.",
    schedule="@daily",
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    tags=["stock-prediction", "task-5.3"],
)
def stock_price_pipeline():

    @task
    def ingest_market_data() -> int:
        WAREHOUSE_DB.parent.mkdir(parents=True, exist_ok=True)

        frames = []
        for ticker in TICKERS:
            data = yf.download(ticker, period="6mo", auto_adjust=True, progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data = data.reset_index()
            data["ticker"] = ticker
            frames.append(data[["Date", "ticker", "Open", "High", "Low", "Close", "Volume"]])

        raw = pd.concat(frames, ignore_index=True)
        raw.columns = [c.lower() for c in raw.columns]

        con = duckdb.connect(str(WAREHOUSE_DB))
        try:
            con.execute("CREATE OR REPLACE TABLE raw_prices AS SELECT * FROM raw")
        finally:
            con.close()

        return len(raw)

    @task
    def transform_with_dbt(row_count: int) -> str:
        result = subprocess.run(
            [
                str(DBT_BIN), "build",
                "--project-dir", str(DBT_PROJECT_DIR),
                "--profiles-dir", str(DBT_PROJECT_DIR),
            ],
            cwd=str(DBT_PROJECT_DIR),  # profiles.yml's warehouse path is relative to this
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise AirflowException(f"dbt build failed:\n{result.stdout}\n{result.stderr}")
        return result.stdout[-3000:]

    @task
    def retrain_models(dbt_log: str) -> str:
        if not DEPLOYMENT_VENV_PYTHON.exists():
            raise AirflowException(
                f"deployment venv not found at {DEPLOYMENT_VENV_PYTHON}. "
                f"Create it with: py -3.12 -m venv deployment_venv && "
                f"deployment_venv/Scripts/pip install -r deployment/requirements.txt"
            )
        result = subprocess.run(
            [str(DEPLOYMENT_VENV_PYTHON), str(TRAIN_SCRIPT)],
            cwd=str(TRAIN_SCRIPT.parent),
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise AirflowException(f"train.py failed:\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}")
        return "retrained"

    row_count = ingest_market_data()
    dbt_log = transform_with_dbt(row_count)
    retrain_models(dbt_log)


stock_price_pipeline()
