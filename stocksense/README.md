# StockSense

StockSense is a soft-computing mini project that combines:
- LSTM neural forecasting
- Genetic Algorithm hyperparameter optimization
- Mamdani Fuzzy Logic signal generation

The project exposes a FastAPI backend and a modern vanilla JS dashboard.

## Project Structure

```text
stocksense/
├── backend/
│   ├── main.py
│   ├── data_pipeline.py
│   ├── lstm_model.py
│   ├── ga_optimizer.py
│   ├── fuzzy_engine.py
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
└── README.md
```

## Setup

1. Open a terminal in `stocksense/backend`.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start FastAPI server:

```bash
uvicorn main:app --reload
```

4. Open `stocksense/frontend/index.html` in a browser.

## API Endpoints

- `GET /api/analyze?ticker=AAPL&run_ga=true`
  - Runs full pipeline (data -> GA -> LSTM -> fuzzy signals)
- `GET /api/predict?ticker=AAPL`
  - Quick prediction path (skips GA, uses default hyperparameters)
- `GET /api/tickers`
  - Returns popular ticker suggestions

All responses include:
- `dates`
- `actual_prices`
- `predicted_prices`
- `signals`
- `ga_history`
- `metrics` (RMSE, MAE)
- `latest_signal`
- `confidence`
- `processing_time_seconds`

## Notes

- GA progress is printed in backend logs generation-by-generation.
- GA route takes longer than quick predict; frontend shows an animated progress bar while waiting.
- This project fetches real stock data from Yahoo Finance via `yfinance`.
