# Portfolio/CV Reference — Time-Series Deep Learning for Stock Markets

> Personal reference doc. Not polished for external readers — pull from this when tailoring a CV bullet, portfolio writeup, or interview story to a specific job posting.

## One-liner
Built an end-to-end deep learning pipeline (LSTM) that forecasts Nasdaq and Vietnam stock prices across multiple horizons, generates buy/sell trading signals from price+technical-indicator patterns, and feeds those forecasts into a Markowitz portfolio optimizer — solo project for CS313 Deep Learning for AI, Fulbright University Vietnam, Spring 2026.

## Elevator pitch variants
- **ML/DS-flavored:** "Designed and trained multiple LSTM architectures for multi-horizon time-series forecasting (1-day, k-day, and k-consecutive-day) on both Nasdaq and Vietnamese equities, including a custom exponentially-decayed loss function to prioritize near-term forecast accuracy."
- **Quant/fintech-flavored:** "Built a signal-to-portfolio pipeline: rule-based technical labeling (SMA/RSI/MACD) → LSTM classifier for entry/exit signals → risk-adjusted company scoring (Sharpe ratio + model MSE) → Markowitz mean-variance optimization via SLSQP to produce concrete portfolio allocations for two investor risk profiles."
- **MLE/engineering-flavored:** "Implemented a reproducible time-series ML workflow with strict train/val/test chronological splitting, walk-forward backtesting, data-leakage checks, and model/artifact versioning via Drive-backed save/load utilities."

## Problem framing
Given historical OHLCV data, answer three escalating questions:
1. What will the price be (next day / k days / next k consecutive days)?
2. Given price + technical indicators, should I buy or sell today?
3. Given a universe of candidate stocks and the model's own forecasts, how should I allocate capital?

## Tech stack
| Layer | Tools |
|---|---|
| Modeling | TensorFlow/Keras — stacked LSTM (also GRU/BiGRU variants defined), Dense heads |
| Data prep | pandas, NumPy, scikit-learn (`MinMaxScaler`, `TimeSeriesSplit`, `train_test_split`) |
| Technical indicators | `ta` library — SMA, RSI, MACD |
| Portfolio math | SciPy `optimize.minimize` (SLSQP), Sharpe ratio, covariance matrices |
| Infra | Google Colab + Google Drive (model/dataset persistence via `joblib`, `.keras` files) |
| (Documented but not present in this repo) | FastAPI + Streamlit deployment, Apache Airflow + Airbyte + dbt orchestration — see "What's not actually built" below |

## Architecture by task

**Task 1 — Nasdaq (single-company, e.g. AAPL), notebook `230105_Task1.ipynb`**
- 6 features (OHLC + Adjusted Close + Volume), `MinMaxScaler`, 60-day window.
- Base model: `LSTM(50, return_seq) → Dropout(0.2) → LSTM(50) → Dropout(0.2) → Dense(25) → Dense(1)`, MSE loss.
- 1.1 next-day / 1.2 k-day-ahead (shift target index by `k-1`) / 1.3 k-consecutive-day (multi-output `Dense(k)`).
- Extras: company filtering (≥120 data points), `TimeSeriesSplit` CV, custom walk-forward backtest (21-day train / 5-day test sliding windows) with an explicit data-leakage assertion.

**Task 2 — Vietnam market (16 banks), notebook `230105_task234.ipynb` cells 0–64**
- Company universe built from a fundamentals funnel: industry filter → founding year ≤2014 & stock rating ≥3.5 → ≥5 years dividend history → P/B > 0.5, P/E < 20, ROA > 0.05 → 16 tickers survive.
- Deeper model: `LSTM(256) → Dropout → LSTM(128) → Dropout → LSTM(64) → Dropout → Dense(32) → Dense(out)`, `AdamW(lr=1e-6)`.
- 2.1 next-day (window=30) / 2.2 k=7-day-ahead (window=60) / 2.3 k=15 consecutive days (window=60).
- **Custom loss** for 2.3: `exponential_decay_weighted_mse` (`@tf.function`, `tf.pow(decay_rate, tf.range(k))`) — weights near-term days more than far-out days instead of treating all 15 output days equally.
- Chronological `train_test_split(shuffle=False)`, per-feature scalers, models cached to Drive as `.keras`, datasets cached as `.npy`/`.pkl`.

**Task 3 — Buy/sell signal classification, cells 65–90**
- Labels generated from rolling 30-day price percentiles (mean-reversion framing): near-bottom → buy, near-top → sell.
- Denoising filter: only train on days where `abs(MACD) > mean(abs(MACD))` for that ticker — excludes flat/sideways days.
- Two independent binary classifiers (`model_buy`, `model_sell`): `LSTM(64) → BatchNorm → LSTM(32) → Dense(10) → Dense(1, sigmoid)`, `binary_crossentropy`, tracked accuracy + precision. Input = 20-day window of normalized Open price.
- Reframes the problem from regression → probabilistic classification (confidence score, not a hard yes/no).

**Task 4 — Portfolio construction, cells 91–103**
- `filter_companies`: for each bank, compute Sharpe ratio + Task-2 model's MSE + 7-day expected return → **Combined Index = `(1 - MSE)*0.4 + Sharpe*0.6`** → rank, exclude weak scorers (e.g. SHB, ACB excluded).
- Markowitz mean-variance optimization: `scipy.optimize.minimize(SLSQP)`, objective = minimize negative Sharpe ratio, constraints = weights sum to 1, bounds [0,1] (no short-selling), covariance matrix via `np.cov` on model-predicted returns.
- Two investor-profile outputs: Prudent (SSB, BID, LPB, OCB — highest combined index/stability) vs. Risk-taking (KLB, OCB, MSB — KLB ~41% expected return despite higher prediction error).

## Technical decisions worth explaining in an interview
- **No shuffling, ever** — `shuffle=False` splits, `TimeSeriesSplit`, manual walk-forward windows, plus an explicit assertion that train indices never overlap/follow test indices. This is the single most-repeated lesson across the report's reflections.
- **Lookback window scales with forecast horizon** — 30 days for next-day, 60 days for 7-/15-day-ahead forecasts (longer horizon needs more historical context).
- **Custom weighted loss for multi-step output** — standard MSE treats day-1 and day-15 of a forecast equally; built an exponential-decay-weighted MSE so near-term (more actionable) predictions are penalized harder.
- **Hybrid rule+learning design (Task 3)** — technical indicators (SMA/RSI/MACD) define labels and a MACD-magnitude filter denoises training data, so the LSTM learns the price "shape" that precedes a rule-confirmed event rather than learning from raw noisy ticks.
- **Fundamentals-first stock universe selection** — before any modeling, filtered candidates by sector, company age, dividend history, and financial ratios (P/B, P/E, ROA) on the theory that model quality is bottlenecked by data quality, not architecture size.
- **Risk-aware capital allocation** — didn't just rank by predicted return; combined model reliability (inverse MSE) with historical risk-adjusted return (Sharpe) before even entering the optimizer, then ran actual quadratic/SLSQP optimization rather than heuristic weighting.

## Task 5.1/5.2 — Deployment (built, verified working end-to-end)
- Location: [deployment/](deployment/). Self-contained on purpose — no Google Drive dependency, unlike Tasks 1–4's notebooks, so anyone can clone the repo and reproduce it.
- **Pipeline (`deployment/train.py`):** pulls 5 years of daily OHLCV for 5 Nasdaq tickers (AAPL, MSFT, GOOGL, AMZN, NVDA) via `yfinance`, trains one LSTM per ticker (2×`LSTM(50)` + dropout, 60-day window, next-day Close), saves model/scaler/backtest artifacts per ticker.
- **Backend (`deployment/api/main.py`, Task 5.1):** FastAPI app that loads all models into memory at startup (not per-request reload) and runs a real forward pass at request time — `GET /tickers`, `GET /predict/{ticker}` (backtest series + live next-day forecast), with an optional `?refresh=true` to pull today's bar from yfinance before predicting.
- **Frontend (`deployment/app/streamlit_app.py`, Task 5.2):** ticker dropdown, actual-vs-predicted Plotly chart, last-close/next-day-prediction/live-inference metric cards.
- **Verified:** trained all 5 tickers successfully, hit the API directly (cached and live-refresh paths, plus 404 handling), and drove the Streamlit UI with Playwright — screenshotted both the default view and a ticker switch (AAPL → NVDA) to confirm the dropdown actually round-trips through the API and re-renders.
- Needed Python ≤3.12 for TensorFlow (this machine defaults to 3.14) — solved with a dedicated `py -3.12` venv (`deployment_venv/`, gitignored), documented in `deployment/README.md`.
- Env quirk worth remembering if this comes up in an interview: **TensorFlow does not yet support Python 3.14** (as of when this was built) — had to explicitly target 3.12 via `py -3.12 -m venv`.

## Known limitations / honest talking points (don't overclaim these on a CV)
- The Task 4 portfolio optimizer tends toward corner solutions (near-100% allocation to one asset) — no max-weight-per-asset constraint was added, a known gap flagged in the report itself.
- The deployment demo predicts next-day **Close** only, using Task 1's simpler architecture (not the deeper Task 2 stack or the k-day/k-consecutive-day variants) — a deliberate scope cut for fast CPU training across 5 tickers, not a limitation of the underlying research.
- `airflow/` (Task 5.3 — Airbyte/dbt orchestration) is still just described in the README's roadmap, not implemented.
- Trained mostly with short epoch counts (5–60) and small-scale experimentation (single-company for Task 1, 16 tickers for Tasks 2–4) — framed as an academic/prototype project, not production-scale.

## Possible CV/portfolio bullets (edit per job posting)
- "Designed and trained LSTM forecasting models across 3 forecast horizons (next-day, k-day-ahead, k-day multi-output) on multi-market equity data, including a custom exponentially-weighted loss function for near-term forecast prioritization."
- "Built a hybrid rules + deep learning trading signal system combining technical indicators (SMA/RSI/MACD) with an LSTM classifier, achieving denoised training via momentum-based sample filtering."
- "Implemented Markowitz portfolio optimization (SciPy SLSQP) driven by deep-learning price forecasts and a custom risk-adjusted scoring function blending model reliability (MSE) and historical Sharpe ratio."
- "Applied rigorous time-series ML practices (chronological splitting, walk-forward backtesting, leakage validation) across a 4-stage forecasting-to-allocation pipeline."
- "Deployed trained models behind a FastAPI inference service (in-memory model caching, live + on-demand data refresh) with a Streamlit dashboard front end, verified end-to-end with automated browser testing (Playwright)."

## Files in this repo
- [README.md](README.md) — project overview / roadmap (Task 5.3/`airflow/` still not built)
- [notebooks/230105_Task1.ipynb](notebooks/230105_Task1.ipynb) — Task 1 (Nasdaq)
- [notebooks/230105_task234.ipynb](notebooks/230105_task234.ipynb) — Tasks 2–4 (Vietnam forecasting, signals, portfolio)
- [report/230105_Report.pdf](report/230105_Report.pdf) — full written reflections per subtask (source for most of the "why" in this doc)
- [deployment/](deployment/) — Task 5.1/5.2, FastAPI + Streamlit (see `deployment/README.md` for run instructions)
- [requirements.txt](requirements.txt) — dependencies for the notebooks; `deployment/requirements.txt` is separate (TF-cpu, yfinance, fastapi, streamlit)
