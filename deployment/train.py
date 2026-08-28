"""
Trains one LSTM per ticker on freshly downloaded Nasdaq OHLCV data and saves
everything the API needs to serve predictions: the model, the feature scaler,
the historical dataframe, and walk-forward test predictions for the
"Actual vs Predicted" chart.

Self-contained on purpose: no dependency on the original Colab / Google Drive
setup, so anyone can clone the repo and reproduce this with `python train.py`.

Usage:
    python train.py
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
FEATURES = ["Open", "High", "Low", "Close", "Volume"]
WINDOW_SIZE = 60
HISTORY_PERIOD = "5y"
TEST_SPLIT = 0.15
EPOCHS = 30
BATCH_SIZE = 32

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")


def build_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def make_sequences(scaled_data, window_size, target_col_idx):
    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i - window_size:i])
        y.append(scaled_data[i, target_col_idx])
    return np.array(X), np.array(y)


def train_ticker(ticker):
    print(f"\n=== {ticker} ===")
    raw = yf.download(ticker, period=HISTORY_PERIOD, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw[FEATURES].dropna()
    raw.index.name = "Date"

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(raw.values)

    close_idx = FEATURES.index("Close")
    X, y = make_sequences(scaled, WINDOW_SIZE, close_idx)

    split = int(len(X) * (1 - TEST_SPLIT))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = build_model((X_train.shape[1], X_train.shape[2]))
    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=[es], verbose=0,
    )

    y_pred_scaled = model.predict(X_test, verbose=0).flatten()
    test_mse = float(np.mean((y_pred_scaled - y_test) ** 2))

    def denorm_close(scaled_close_values):
        dummy = np.zeros((len(scaled_close_values), len(FEATURES)))
        dummy[:, close_idx] = scaled_close_values
        return scaler.inverse_transform(dummy)[:, close_idx]

    actual_prices = denorm_close(y_test)
    predicted_prices = denorm_close(y_pred_scaled)
    test_dates = raw.index[WINDOW_SIZE + split: WINDOW_SIZE + len(X)]

    # One extra forecast: predict the very next trading day beyond all known data.
    last_window = scaled[-WINDOW_SIZE:].reshape(1, WINDOW_SIZE, len(FEATURES))
    next_day_scaled = model.predict(last_window, verbose=0).flatten()
    next_day_price = float(denorm_close(next_day_scaled)[0])

    ticker_dir = os.path.join(ARTIFACTS_DIR, ticker)
    os.makedirs(ticker_dir, exist_ok=True)

    model.save(os.path.join(ticker_dir, "model.keras"))
    joblib.dump(scaler, os.path.join(ticker_dir, "scaler.pkl"))
    raw.to_csv(os.path.join(ticker_dir, "history.csv"))

    backtest_df = pd.DataFrame({
        "date": test_dates.strftime("%Y-%m-%d"),
        "actual": actual_prices,
        "predicted": predicted_prices,
    })
    backtest_df.to_csv(os.path.join(ticker_dir, "backtest.csv"), index=False)

    metadata = {
        "ticker": ticker,
        "features": FEATURES,
        "window_size": WINDOW_SIZE,
        "test_mse": test_mse,
        "last_close": float(raw["Close"].iloc[-1]),
        "last_date": raw.index[-1].strftime("%Y-%m-%d"),
        "next_day_prediction": next_day_price,
        "n_history_rows": len(raw),
    }
    with open(os.path.join(ticker_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  test MSE (scaled): {test_mse:.6f} | last close: {metadata['last_close']:.2f} "
          f"| next-day prediction: {next_day_price:.2f}")
    return metadata


def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    all_metadata = [train_ticker(t) for t in TICKERS]
    with open(os.path.join(ARTIFACTS_DIR, "tickers.json"), "w") as f:
        json.dump(all_metadata, f, indent=2)
    print(f"\nSaved artifacts for {len(all_metadata)} tickers to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
