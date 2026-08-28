# Deployment — Task 5.1 & 5.2

A self-contained FastAPI + Streamlit deployment of the LSTM forecasting work from
[`../notebooks`](../notebooks). Unlike the notebooks (which trained against Google
Drive-hosted data in Colab), everything here runs from a fresh `yfinance` pull, so
it's reproducible on any machine.

## Architecture

```
yfinance  ──▶  train.py  ──▶  artifacts/<TICKER>/{model.keras, scaler.pkl,
                                              history.csv, backtest.csv, metadata.json}
                                        │
                                        ▼
                              api/main.py  (FastAPI, loads models into memory,
                                             runs real inference per request)
                                        │
                                        ▼
                          app/streamlit_app.py  (ticker picker, actual-vs-
                                                   predicted chart, live metrics)
```

## Setup

TensorFlow needs Python ≤3.12 (this project's environment defaults to a newer
Python that isn't yet supported). A dedicated venv is used:

```bash
py -3.12 -m venv ../deployment_venv
../deployment_venv/Scripts/pip install -r requirements.txt
```

## Run it

1. **Train** (fetches 5y of daily OHLCV for `AAPL, MSFT, GOOGL, AMZN, NVDA` via
   yfinance and trains one LSTM per ticker):
   ```bash
   ../deployment_venv/Scripts/python train.py
   ```
   Takes a few minutes on CPU. Produces `artifacts/<TICKER>/*` and `artifacts/tickers.json`.

2. **API**:
   ```bash
   ../deployment_venv/Scripts/uvicorn api.main:app --reload --port 8000
   ```
   - `GET /tickers` — list trained tickers with test MSE / last known price.
   - `GET /predict/{ticker}` — actual-vs-predicted backtest series + a next-day
     forecast computed live from the loaded model. Add `?refresh=true` to
     re-pull today's bar from yfinance before forecasting.

3. **Frontend**:
   ```bash
   ../deployment_venv/Scripts/streamlit run app/streamlit_app.py
   ```
   Opens a browser UI: pick a ticker, see the actual-vs-predicted chart,
   the model's test MSE, and a live next-day price forecast.

## Notes / honest limitations

- Model architecture matches Task 1's baseline (2×`LSTM(50)` + dropout), not the
  deeper Task 2 architecture — chosen for fast CPU training across 5 tickers.
- Predicts next-day **Close** only (single horizon), not the k-day or
  k-consecutive-day variants from the notebooks.
- `refresh=true` inference depends on yfinance being reachable at request time;
  it silently falls back to the cached snapshot on failure.
- No auth, rate limiting, or persistence layer — this is a portfolio-grade demo,
  not a production service.
