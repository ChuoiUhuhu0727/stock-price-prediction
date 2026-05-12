# Time-series Data and Application to Stock Markets
### CS313 Deep Learning for Artificial Intelligence — Spring 2026

## 🧭 Navigation
* **Tasks 1-4:** [`./notebooks`](./notebooks) directory
* **Task 5.1 & 5.2 (Deployment):** [`./deployment`](./deployment) directory
* **Task 5.3 (Automation):** [`./airflow`](./airflow) directory
* **Task 6 (Documentation):** [`./report`](./report) directory

## 📝 Overview
This project explores the use of deep learning to forecast stock market trends and generate actionable financial insights. By analyzing Nasdaq and Vietnam stock data, I developed predictive models using multi-feature inputs, identified trading signals, and simulated portfolio performance. The project culminates in an industry-standard engineering workflow, including automated data pipelines with Apache Airflow and model deployment as a SaaS platform.

## 📊 Table of Contents
1. [Introduction](#introduction)
2. [Key Features](#key-features)
3. [Project Workflow](#project-workflow)
   - [Task 1: Nasdaq Stock Price Prediction](#task-1-nasdaq-stock-price-prediction)
   - [Task 2: Vietnam Stock Price Prediction](#task-2-vietnam-stock-price-prediction)
   - [Task 3: Trading Signal Identification](#task-3-trading-signal-identification-for-vietnam-market)
   - [Task 4: Risk and Return Analysis](#task-4-risk-and-return-analysis-across-industries)
   - [Task 5: Industry Standard Deployment](#task-5-industry-standard-for-deployment-and-ease-of-use)
4. [Lessons Learned](#lessons-learned)
5. [Future Improvements](#future-improvements)
6. [Conclusion](#conclusion)

---

## 🚀 Introduction
In today’s volatile socio-economic climate, the ability to analyze and forecast dynamic trends is essential for data-driven decision-making. This report explores the application of **deep learning** to time-series forecasting within global and local stock markets. By leveraging historical data from both the Nasdaq and Vietnam stock exchanges, this project evaluates how multi-feature models can capture complex temporal patterns. 

My approach follows a structured path through six key objectives: beginning with predictive modeling, moving toward trading signal identification, and culminating in portfolio optimization and automated deployment. This repository chronicles my technical journey, detailing the methodology, experimental results, and the deep learning architectures used to derive practical market solutions.

## 🛠️ Key Features
* **Predictive Models:** Built using **LSTM (Long Short-Term Memory)** networks and **CNNs** for temporal feature extraction.
* **Multi-Feature Input:** Integrated six key features (Open, High, Low, Close, Adjusted Close, and Volume) to provide richer context for predictions.
* **Portfolio Optimization:** Applied mathematical **Quadratic Programming** and the **Sharpe Ratio** to optimize asset weights and minimize risk.
* **SaaS Deployment:** Deployed models as a REST API using **FastAPI** and created a user-friendly frontend with **Streamlit**.
* **Automated Workflow:** Built an **Apache Airflow** DAG to orchestrate data fetching, prediction, and database updates.

---

## 📈 Project Workflow

### Task 1: Nasdaq Stock Price Prediction
* **Goal:** Develop a robust predictive model for international tech stocks.
* **Process:** * Upgraded the model input from a single feature (Open price) to **six features** (OHLCV + Adjusted Close).
    * Updated input shapes from `(timesteps, 1)` to `(timesteps, 6)` and applied `MinMaxScaler` across all features.
    * Implemented **Recursive Strategy** for multi-day forecasting, where the model uses its own predictions as future inputs.
* **Outcome:** Improved model context, allowing the LSTM to detect patterns between volume spikes and price volatility.

### Task 2: Vietnam Stock Price Prediction
* **Goal:** Adapt the predictive architecture to the localized dynamics of the Vietnam market.
* **Process:** * Fetched data using the `vnstock` library for major tickers (e.g., VIC, VNM).
    * Addressed the "Lag Effect" by predicting **Price Differences (Diff)** instead of absolute prices, forcing the model to learn movement magnitude rather than just mimicking the current price.
* **Outcome:** Validated the transferability of the LSTM architecture across different market environments.

### Task 3: Trading Signal Identification for Vietnam Market
* **Goal:** Classify time-series segments into actionable "Buy," "Sell," or "Hold" signals.
* **Process:** * Combined LSTM predictions with technical indicators like **SMA (Simple Moving Average)** and **RSI (Relative Strength Index)**.
    * Defined signals based on the probability of a price being at a local minimum/maximum within a look-back window.
* **Outcome:** Created a balanced signal generator that filters market noise through technical indicator thresholds.

### Task 4: Risk and Return Analysis Across Industries
* **Goal:** Evaluate performance metrics and optimize asset allocation across sectors.
* **Process:** * Calculated **Expected Returns** and **Covariance Matrices** for diversified portfolios.
    * Moved from simple ranking to **Numerical Optimization** using `scipy.optimize.minimize` (SLSQP method).
    * Targeted the highest possible **Sharpe Ratio** for risk-adjusted returns.
* **Outcome:** Demonstrated that mathematical optimization significantly outperforms heuristic asset selection by reducing correlation-based risks.

### Task 5: Industry Standard for Deployment and Ease of Use
#### 5.1 & 5.2 Model as API & SaaS
* **Goal:** Provide accessible predictions through a professional web interface.
* **Implementation:** Deployed a **FastAPI** backend to serve model inferences. The **Streamlit** frontend allows users to select tickers and visualize "Actual vs. Predicted" price segments in real-time.

#### 5.3 Orchestrating a Complete Workflow
* **Goal:** Automate the AI lifecycle.
* **Implementation:** Designed an **Airflow DAG** that schedules daily data ingestion via **Airbyte**, transforms data using **dbt**, and triggers model updates to ensure the SaaS platform always displays current forecasts.

---

## 🎓 Lessons Learned
* **Temporal Integrity:** Preserving chronological order (No Shuffling) is non-negotiable in time-series validation to avoid data leakage.
* **Dimensionality:** Moving to a multi-feature input (6 features) drastically improves the model's "environmental awareness" of the market.
* **The Lag Effect:** Predicting absolute prices often leads to models that simply "shadow" the previous day's price. Predicting **Price Change (Diff)** is a more effective strategy for learning real patterns.
* **Garbage In, Garbage Out:** Portfolio optimization is highly sensitive to the accuracy of the underlying LSTM predictions.

## 🔮 Future Improvements
* **Transformer Architectures:** Implementing Self-Attention mechanisms to capture longer-term dependencies than standard LSTMs.
* **Sentiment Analysis:** Integrating NLP to process financial news and social trends as additional model features.
* **GPU Scaling:** Optimizing the Airflow pipeline to utilize GPU-accelerated training for larger ensembles.

## 📌 Conclusion
This project served as a comprehensive journey through the AI lifecycle—from raw data preprocessing and deep learning research to mathematical optimization and professional deployment. It highlights the power of combining data engineering with predictive analytics to turn market volatility into actionable financial intelligence.
