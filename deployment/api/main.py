"""
FastAPI backend serving the LSTM stock-price forecasts trained by train.py.

Loads each ticker's model + scaler into memory once at startup and runs real
inference per request (not just replaying precomputed numbers), optionally
pulling fresh OHLCV data from yfinance so the "next day" forecast is computed
on the latest available bar.

Run (from deployment/):
    uvicorn api.main:app --reload --port 8000
"""
import json
import os
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tensorflow.keras.models import load_model

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts")

MODEL_CACHE: dict[str, dict] = {}


def _load_ticker_artifacts(ticker: str) -> dict:
    tdir = os.path.join(ARTIFACTS_DIR, ticker)
    with open(os.path.join(tdir, "metadata.json")) as f:
        metadata = json.load(f)
    return {
        "model": load_model(os.path.join(tdir, "model.keras")),
        "scaler": joblib.load(os.path.join(tdir, "scaler.pkl")),
        "history": pd.read_csv(os.path.join(tdir, "history.csv"), index_col="Date", parse_dates=True),
        "backtest": pd.read_csv(os.path.join(tdir, "backtest.csv")),
        "metadata": metadata,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    index_path = os.path.join(ARTIFACTS_DIR, "tickers.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            for entry in json.load(f):
                ticker = entry["ticker"]
                MODEL_CACHE[ticker] = _load_ticker_artifacts(ticker)
        print(f"Loaded {len(MODEL_CACHE)} models into memory: {list(MODEL_CACHE)}")
    else:
        print("WARNING: no artifacts/tickers.json found — run train.py first.")
    yield
    MODEL_CACHE.clear()


app = FastAPI(title="Stock Price Prediction API", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TickerSummary(BaseModel):
    ticker: str
    test_mse: float
    last_close: float
    last_date: str
    n_history_rows: int


class PredictionResponse(BaseModel):
    ticker: str
    dates: list[str]
    actual: list[float]
    predicted: list[float]
    next_day_prediction: float
    last_close: float
    last_date: str
    test_mse: float
    live_inference: bool


def _get_cached(ticker: str) -> dict:
    ticker = ticker.upper()
    if ticker not in MODEL_CACHE:
        raise HTTPException(
            status_code=404,
            detail=f"No trained model for ticker '{ticker}'. Run train.py first, or check /tickers.",
        )
    return MODEL_CACHE[ticker]


def _predict_next_day(entry: dict, history: pd.DataFrame) -> float:
    """Runs an actual forward pass through the cached Keras model."""
    features = entry["metadata"]["features"]
    window_size = entry["metadata"]["window_size"]
    scaler = entry["scaler"]

    recent = history[features].tail(window_size).values
    scaled = scaler.transform(recent).reshape(1, window_size, len(features))

    pred_scaled = entry["model"].predict(scaled, verbose=0).flatten()
    close_idx = features.index("Close")
    dummy = np.zeros((1, len(features)))
    dummy[:, close_idx] = pred_scaled
    return float(scaler.inverse_transform(dummy)[:, close_idx][0])


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(MODEL_CACHE)}


@app.get("/tickers", response_model=list[TickerSummary])
def list_tickers():
    if not MODEL_CACHE:
        raise HTTPException(status_code=503, detail="No trained artifacts found. Run train.py first.")
    return [
        {
            "ticker": t,
            "test_mse": e["metadata"]["test_mse"],
            "last_close": e["metadata"]["last_close"],
            "last_date": e["metadata"]["last_date"],
            "n_history_rows": e["metadata"]["n_history_rows"],
        }
        for t, e in MODEL_CACHE.items()
    ]


@app.get("/predict/{ticker}", response_model=PredictionResponse)
def predict(ticker: str, refresh: bool = False):
    """
    Returns the backtest (actual vs predicted on held-out test data, for
    charting) plus a next-day forecast computed live from the trained model.

    `refresh=true` re-pulls the latest bars from yfinance before forecasting,
    so the prediction reflects today's data instead of train.py's snapshot.
    """
    entry = _get_cached(ticker)
    metadata = entry["metadata"]
    history = entry["history"]
    live_inference = False

    if refresh:
        try:
            fresh = yf.download(ticker.upper(), period="6mo", auto_adjust=True, progress=False)
            if isinstance(fresh.columns, pd.MultiIndex):
                fresh.columns = fresh.columns.get_level_values(0)
            fresh = fresh[metadata["features"]].dropna()
            fresh.index.name = "Date"
            if not fresh.empty:
                history = fresh
                live_inference = True
        except Exception:
            pass  # fall back to cached snapshot below

    next_day_prediction = _predict_next_day(entry, history)
    backtest = entry["backtest"]

    return PredictionResponse(
        ticker=ticker.upper(),
        dates=backtest["date"].tolist(),
        actual=backtest["actual"].tolist(),
        predicted=backtest["predicted"].tolist(),
        next_day_prediction=next_day_prediction,
        last_close=float(history["Close"].iloc[-1]),
        last_date=str(history.index[-1].date()) if hasattr(history.index[-1], "date") else str(history.index[-1]),
        test_mse=metadata["test_mse"],
        live_inference=live_inference,
    )
