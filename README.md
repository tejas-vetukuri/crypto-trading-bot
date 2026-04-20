# Crypto Trading Bot (FYP)

This project implements a hybrid machine learning trading system that combines supervised learning (LSTM and XGBoost) with reinforcement learning for cryptocurrency signal generation and trade filtering.

## Features
- Data acquisition from Binance (OHLCV) and Bybit (derivatives data)
- Feature engineering with technical indicators and alternative features
- LSTM and XGBoost models for price direction prediction
- Weighted ensemble of model outputs
- Reinforcement learning (Q-learning) trade filtering
- Streamlit dashboard for live signals and evaluation
- Backtesting and evaluation framework
- Unit and integration testing

---

## Repository Structure

The repository is structured into the following main components:

- **Data Module (`data/`)**
  - API wrappers for market data retrieval
  - Feature engineering pipelines
  - Technical indicator computation
  - Integration of alternative features

- **Model Module (`models/`)**
  - `lstm/`: sequence construction, training, evaluation, thresholding, optimisation
  - `xgboost/`: feature-based training, thresholding, evaluation, optimisation
  - `rl/`: weighted ensemble, Q-learning training, evaluation (trade simulation), trade configuration optimisation
  - `baselines/`: random, majority, Random Forest, Simple RNN, and evaluation
  - `alternate/`: experimental models (e.g., GNN, alternative targets)
  - `saved/`: present within each model sub-module, storing trained models, scalers, metrics, and predictions for reuse and live inference

- **Frontend Module (`frontend/`)**
  - Streamlit-based user interface
  - Pages for live signal generation and evaluation
  - Service layer connecting backend components for data handling, feature construction, model loading, and inference

- **Testing Module (`tests/`)**
  - Unit and integration tests implemented using pytest

- **Scripts (`scripts/`)**
  - Auxiliary scripts for plotting, debugging, and exploratory testing

---

## Setup and Run

Install requirements:
```bash
pip install -r requirements.txt
```

Launch Streamlit Dashboard:
```bash
streamlit run frontend/app.py
```
