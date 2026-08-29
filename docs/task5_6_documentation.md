# Task 5 & 6 — Deployment, Orchestration, Documentation

This extends the original report ([`report/230105_Report.pdf`](../report/230105_Report.pdf),
which covers Tasks 1–4) with the engineering work for Tasks 5.1–5.3, and
serves as Task 6 (Documentation) for that work.

## System architecture

```
┌──────────────┐     ┌──────────────────┐     ┌────────────────────┐
│ yfinance      │ ──▶ │ ingest_market_data│ ──▶ │  raw_prices          │
│ (data source) │     │ (Airflow task)    │     │  (DuckDB, main.*)    │
└──────────────┘     └──────────────────┘     └────────────────────┘
                                                          │
                                                          ▼
                                                ┌────────────────────┐
                                                │ dbt build            │
                                                │ staging.stg_prices    │
                                                │ marts.daily_returns   │
                                                │ marts.latest_snapshot │
                                                └────────────────────┘
                                                          │
                                                          ▼
                                                ┌────────────────────┐
                                                │ deployment/train.py   │
                                                │ (retrain_models task) │
                                                └────────────────────┘
                                                          │
                                                          ▼
                                       ┌───────────────────────────────┐
                                       │ deployment/artifacts/<TICKER>/   │
                                       │  model.keras, scaler.pkl, ...    │
                                       └───────────────────────────────┘
                                                          │
                                                          ▼
                          ┌──────────────────┐     ┌───────────────────┐
                          │ Streamlit frontend │◀──▶│ FastAPI backend      │
                          │ (Task 5.2)         │    │ (Task 5.1)            │
                          └──────────────────┘     └───────────────────┘
```

Three independently-runnable pieces, connected by files and HTTP — not one
monolithic script:
- **[`deployment/`](../deployment/)** (Task 5.1/5.2) — trains models, serves
  them over HTTP, renders them in a browser.
- **[`airflow/`](../airflow/)** (Task 5.3) — schedules the data refresh that
  keeps `deployment/`'s models current.
- They share one contract: `deployment/train.py` writes to
  `deployment/artifacts/`, and that's the only thing the orchestration layer
  needs to know about the serving layer.

## Task 5.1 — API (`deployment/api/main.py`)

FastAPI app that loads every trained model into memory once at startup
(`lifespan` context manager) instead of per-request — a `.keras` load is
too slow to repeat on every call. `GET /predict/{ticker}` runs a real
forward pass through the cached model on each request; `?refresh=true`
re-pulls the latest bar from yfinance first, so the forecast can reflect
today's market instead of train-time data.

**Why this design:** the goal was to avoid the two easy-but-dishonest
shortcuts — (a) precomputing everything at train time and just replaying
JSON (not "serving a model," just serving a file), or (b) reloading the
model from disk on every request (technically live, but doesn't scale and
misrepresents what a real inference service looks like).

## Task 5.2 — Frontend (`deployment/app/streamlit_app.py`)

Plain HTTP client of the API above — ticker dropdown, actual-vs-predicted
Plotly chart, live metrics. Deliberately has zero knowledge of TensorFlow,
model files, or scalers; it only knows the API's JSON contract. That
separation is what makes it possible to swap the frontend (or add a second
client) without touching the model-serving code at all.

## Task 5.3 — Orchestration (`airflow/`)

A real Apache Airflow DAG (`airflow/dags/stock_pipeline_dag.py`, TaskFlow
API) with three tasks in sequence:

1. **`ingest_market_data`** — pulls fresh OHLCV bars via `yfinance`, lands
   them in a `raw_prices` table in a local DuckDB warehouse.
2. **`transform_with_dbt`** — runs a real dbt project (`airflow/dbt_stock/`)
   against that warehouse: a staging layer that cleans/types the raw rows,
   and a marts layer that computes daily returns, 20-day rolling
   mean/volatility, and a latest-per-ticker snapshot table — plus 7
   `not_null`/`unique` data tests that fail the run if the data shape ever
   breaks.
3. **`retrain_models`** — shells out to `deployment/train.py` in the
   separate `deployment_venv` (where TensorFlow lives), so a full pipeline
   run leaves `deployment/artifacts/` refreshed and ready for the API to
   pick up.

### Design decisions worth explaining in an interview

- **Two separate venvs** (`airflow_venv`, `deployment_venv`). Airflow's
  dependency pins and TensorFlow's don't coexist cleanly, and dbt-duckdb
  pulled in `protobuf>=6.0` while Airflow's OpenTelemetry integration wants
  `<5.0` — a real conflict, sidestepped by keeping training in its own
  environment and having Airflow *orchestrate* it via subprocess rather than
  *import* it. This is the same pattern real MLOps setups use when a
  heavyweight training stack shouldn't live inside the scheduler's process.
- **`retrain_models` runs the model training in a subprocess, not a Python
  import** — mirrors how you'd trigger training in a separate container/pod
  in a real deployment, and keeps Airflow's own environment lightweight.
- **The "Airbyte" gap, named honestly.** The README's original roadmap named
  Airbyte for ingestion. Airbyte is a multi-container ELT platform (its own
  server, UI, workers, backing Postgres) — standing that up wouldn't do
  anything here that a direct `yfinance` pull doesn't already do for 5
  tickers, and isn't practical for a local portfolio demo. `ingest_market_data`
  is the honest, runnable stand-in with the same contract (land raw data for
  dbt to build on). **dbt and Airflow are the real tools**, installed and run,
  not mocked — only the ingestion connector is simplified.

### What was actually verified (not just "the code looks right")

- `dbt build` run directly against the warehouse: **3 models, 7 data tests,
  all PASS** (`stg_prices` view; `daily_returns` and `latest_snapshot`
  tables).
- `airflow dags list-import-errors`: DAG parses cleanly, **zero import
  errors**.
- `airflow dags test stock_price_pipeline <date>`: full local run of all
  three tasks in order, without needing the scheduler/webserver — the
  standard way to validate an Airflow DAG end-to-end.
- **Known environment caveat:** Apache Airflow does not officially support
  native Windows (POSIX-only; Windows support is tracked upstream but not a
  priority for the project). It ran here with a non-fatal warning (a failed
  attempt to symlink the "latest" log directory, which Windows restricts by
  default). For a real deployment, Airflow would run under WSL2, Docker, or
  a Linux host — exactly what the Apache project itself recommends. This
  repo's `airflow/README.md` documents both paths.

## Honest limitations (Task 5 & 6 scope)

- The retraining task calls the *whole* `train.py` (all 5 tickers, full
  epoch budget) — no incremental/warm-start training. Fine for a daily
  batch job at this scale; would need rethinking at higher ticker counts.
- No orchestration-level model validation gate (e.g., "don't promote a
  retrained model if its test MSE regressed") — `retrain_models` succeeds
  as long as `train.py` exits 0, regardless of whether the new model is
  actually better than the one it replaces.
- No containerization (Docker/Kubernetes) — everything here runs as local
  processes against local venvs. That's the natural "what's next" if this
  were pushed toward production rather than a portfolio demo.
