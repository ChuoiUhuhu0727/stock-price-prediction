# Orchestration — Task 5.3

Real Apache Airflow DAG + real dbt project, orchestrating a daily
ingest -> transform -> retrain cycle so the FastAPI service in
[`../deployment`](../deployment) can serve fresh forecasts.

```
      ┌─────────────────────┐     ┌──────────────────┐     ┌───────────────────┐
      │ ingest_market_data   │ ──▶ │ transform_with_dbt│ ──▶ │  retrain_models     │
      │ (yfinance -> DuckDB) │     │ (dbt build)        │     │ (deployment/train.py)│
      └─────────────────────┘     └──────────────────┘     └───────────────────┘
        stands in for Airbyte        staging -> marts          runs in deployment_venv
```

## Honest scope note

The README's original roadmap names **Airbyte** for ingestion. Airbyte is a
multi-container ELT platform (its own server, UI, workers, Postgres/Temporal
backing services) — standing up the full platform isn't practical for a
local portfolio demo, and it wouldn't do anything here that a direct
`yfinance` pull doesn't already do for 5 tickers. `ingest_market_data` is the
honest, runnable stand-in: same role in the pipeline (land raw data into the
warehouse), same downstream contract (a `raw_prices` table dbt can build on).
**dbt and Airflow themselves are the real thing** — installed, run, and
verified below, not mocked.

## Layout

```
airflow/
  dags/stock_pipeline_dag.py   # the DAG (TaskFlow API)
  dbt_stock/                   # dbt project (staging + marts models, tests)
  warehouse/                   # stock.duckdb gets created here on first run (gitignored)
```

## Setup

Airflow needs its own venv — its dependency pins don't mix well with
TensorFlow's (in `deployment_venv`):

```bash
py -3.12 -m venv airflow_venv
airflow_venv/Scripts/pip install "apache-airflow==2.10.4" \
  --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.10.4/constraints-3.12.txt"
airflow_venv/Scripts/pip install dbt-duckdb yfinance duckdb
```

`deployment_venv` (see [`../deployment/README.md`](../deployment/README.md))
must also exist, since `retrain_models` shells out to it.

## Run it

**Test a single full DAG run without standing up the scheduler/webserver**
(this is the standard way to validate a DAG locally):

```bash
set AIRFLOW_HOME=%cd%\airflow_home
airflow_venv/Scripts/airflow.exe db migrate
airflow_venv/Scripts/airflow.exe dags test stock_price_pipeline 2026-01-01
```

**Run the dbt project on its own** (useful while iterating on models,
without re-pulling data or retraining):

```bash
airflow_venv/Scripts/python -m dbt build --project-dir dbt_stock --profiles-dir dbt_stock
```

**Inspect the warehouse:**

```bash
airflow_venv/Scripts/python -c "import duckdb; print(duckdb.connect('airflow/warehouse/stock.duckdb').sql('select * from latest_snapshot').df())"
```

**Full scheduler + webserver** (only if you actually want the Airflow UI):

```bash
set AIRFLOW_HOME=%cd%\airflow_home
airflow_venv/Scripts/airflow.exe standalone
```

## What each dbt model does

- `staging/stg_prices.sql` — casts/cleans the raw landed rows, drops nulls.
- `marts/daily_returns.sql` — per-ticker daily return, 20-day rolling mean
  price, 20-day rolling volatility (stdev of returns).
- `marts/latest_snapshot.sql` — one row per ticker: the most recent day's
  metrics. This is the table a dashboard or downstream service would query
  for "current state."

`schema.yml` files add `not_null`/`unique` tests on the key columns —
`dbt build` runs the models *and* these tests in one command, so a broken
upstream data change fails the DAG instead of silently propagating.
