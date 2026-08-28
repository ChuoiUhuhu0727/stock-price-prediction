"""
Streamlit frontend for the Stock Price Prediction API.

Run (from deployment/), with the API already running on :8000:
    streamlit run app/streamlit_app.py
"""
import os

import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.environ.get("STOCK_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Stock Price Prediction", page_icon="📈", layout="wide")
st.title("📈 Stock Price Prediction — LSTM")
st.caption("Actual vs. predicted closing price on held-out test data, plus a live next-day forecast from the trained model.")


@st.cache_data(ttl=60)
def fetch_tickers():
    resp = requests.get(f"{API_URL}/tickers", timeout=10)
    resp.raise_for_status()
    return resp.json()


def fetch_prediction(ticker: str, refresh: bool):
    resp = requests.get(f"{API_URL}/predict/{ticker}", params={"refresh": refresh}, timeout=30)
    resp.raise_for_status()
    return resp.json()


try:
    tickers = fetch_tickers()
except requests.exceptions.RequestException as e:
    st.error(
        f"Can't reach the API at `{API_URL}`. Make sure it's running "
        f"(`uvicorn api.main:app --reload` from the `deployment/` folder) and that "
        f"`train.py` has been run at least once.\n\nDetails: {e}"
    )
    st.stop()

if not tickers:
    st.warning("No trained tickers found. Run `python train.py` first.")
    st.stop()

ticker_options = {t["ticker"]: t for t in tickers}

with st.sidebar:
    st.header("Settings")
    selected_ticker = st.selectbox("Ticker", list(ticker_options.keys()))
    live_refresh = st.checkbox(
        "Pull latest market data before forecasting",
        value=False,
        help="Calls yfinance for fresh bars before running inference. Off by default for speed/reliability.",
    )
    st.divider()
    info = ticker_options[selected_ticker]
    st.metric("Test MSE (scaled)", f"{info['test_mse']:.6f}")
    st.caption(f"Trained on {info['n_history_rows']} trading days, through {info['last_date']}")

data = fetch_prediction(selected_ticker, live_refresh)

col1, col2, col3 = st.columns(3)
col1.metric("Last close", f"${data['last_close']:.2f}", help=f"as of {data['last_date']}")
delta = data["next_day_prediction"] - data["last_close"]
col2.metric("Next-day prediction", f"${data['next_day_prediction']:.2f}", f"{delta:+.2f}")
col3.metric("Live inference", "✅ fresh data" if data["live_inference"] else "cached snapshot")

fig = go.Figure()
fig.add_trace(go.Scatter(x=data["dates"], y=data["actual"], name="Actual", mode="lines", line=dict(color="#2563eb")))
fig.add_trace(go.Scatter(x=data["dates"], y=data["predicted"], name="Predicted", mode="lines", line=dict(color="#f97316", dash="dash")))
fig.update_layout(
    title=f"{selected_ticker} — Actual vs Predicted Close (test set)",
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Model: 2-layer LSTM (50 units each) on 60-day OHLCV windows, trained per-ticker on 5 years of Nasdaq data. "
    "Source: [train.py](../train.py)."
)
