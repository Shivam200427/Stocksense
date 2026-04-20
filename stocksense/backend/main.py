"""FastAPI entrypoint for StockSense.

Routes expose a full soft-computing pipeline (LSTM + GA + Fuzzy) and a quick mode.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Dict, List

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from data_pipeline import inverse_close, prepare_data
from fuzzy_engine import batch_signals
from ga_optimizer import optimize_hyperparameters
from lstm_model import estimate_confidence, train_and_evaluate


POPULAR_TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "TSLA",
    "NVDA",
    "META",
    "NFLX",
    "RELIANCE.NS",
    "TCS.NS",
]

DEFAULT_PARAMS = {
    "lstm_units": 64,
    "dropout_rate": 0.2,
    "learning_rate": 1e-3,
    "lookback_window": 60,
}

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "stocksense_artifacts"


app = FastAPI(title="StockSense API", version="1.0.0")

# CORS allows the vanilla frontend to call backend endpoints from a different port.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _ticker_to_file_key(ticker: str) -> str:
    return ticker.upper().replace(".", "_")


def _file_key_to_ticker(file_key: str) -> str:
    if file_key.endswith("_NS"):
        return file_key[:-3] + ".NS"
    return file_key


def _latest_artifact_file(pattern: str) -> Path | None:
    matches = sorted(ARTIFACTS_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def _compute_macd_signal(series: pd.Series) -> pd.Series:
    ema_fast = series.ewm(span=12, adjust=False).mean()
    ema_slow = series.ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=9, adjust=False).mean()
    return signal.fillna(0.0)


def _load_artifact_response(ticker: str) -> Dict | None:
    if not ARTIFACTS_DIR.exists():
        return None

    key = _ticker_to_file_key(ticker)
    pred_path = ARTIFACTS_DIR / "predictions" / f"{key}_predictions.csv"
    if not pred_path.exists():
        return None

    pred_df = pd.read_csv(pred_path)
    required_cols = {"Date", "Actual_Close", "Predicted_Close"}
    if not required_cols.issubset(pred_df.columns):
        return None

    pred_df = pred_df.dropna(subset=["Date", "Actual_Close", "Predicted_Close"]).copy()
    if pred_df.empty:
        return None

    actual = pred_df["Actual_Close"].astype(float).to_numpy()
    predicted = pred_df["Predicted_Close"].astype(float).to_numpy()
    dates = pred_df["Date"].astype(str).tolist()
    last_date = pd.to_datetime(dates[-1], errors="coerce")
    next_trading_date = (
        str((last_date + BDay(1)).date()) if pd.notna(last_date) else dates[-1]
    )

    # Build RSI/MACD from predicted track to keep frontend charts and fuzzy inputs aligned.
    pred_series = pd.Series(predicted)
    rsi_values = _compute_rsi(pred_series).astype(float).tolist()
    macd_values = _compute_macd_signal(pred_series).astype(float).tolist()

    price_change_pct = (((predicted - actual) / np.maximum(actual, 1e-8)) * 100.0).tolist()
    try:
        fuzzy = batch_signals(price_change_pct, rsi_values, macd_values)
    except Exception:
        # Fallback: deterministic heuristic labels when fuzzy engine has sparse edge failures.
        scores: List[float] = []
        labels: List[str] = []
        confidences: List[float] = []
        for pct in price_change_pct:
            score = float(np.clip(50.0 + (pct * 2.5), 0.0, 100.0))
            if score >= 80:
                label = "Strong Buy"
            elif score >= 60:
                label = "Buy"
            elif score >= 40:
                label = "Hold"
            elif score >= 20:
                label = "Sell"
            else:
                label = "Strong Sell"
            scores.append(score)
            labels.append(label)
            confidences.append(float(min(100.0, abs(score - 50.0) * 2.0)))
        fuzzy = {"scores": scores, "labels": labels, "confidences": confidences}

    rmse = float(np.sqrt(np.mean((predicted - actual) ** 2)))
    mae = float(np.mean(np.abs(predicted - actual)))

    metrics_path = _latest_artifact_file("multi_company_metrics_*.csv")
    if metrics_path is not None:
        try:
            metrics_df = pd.read_csv(metrics_path)
            row = metrics_df.loc[metrics_df["ticker"].astype(str).str.upper() == ticker.upper()]
            if not row.empty:
                rmse = float(row.iloc[0].get("rmse", rmse))
                mae = float(row.iloc[0].get("mae", mae))
        except Exception:
            pass

    summary_path = _latest_artifact_file("summary_*.json")
    summary = {}
    if summary_path is not None:
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            summary = {}

    mean_fuzzy_conf = float(np.mean(fuzzy["confidences"])) if fuzzy["confidences"] else 0.0

    return {
        "ticker": ticker.upper(),
        "hyperparameters": {**DEFAULT_PARAMS, "source": "pretrained_artifacts"},
        "dates": dates,
        "actual_prices": [float(x) for x in actual.tolist()],
        "predicted_prices": [float(x) for x in predicted.tolist()],
        "rsi": rsi_values,
        "macd": macd_values,
        "price_change_pct": [float(x) for x in price_change_pct],
        "signals": fuzzy["labels"],
        "signal_scores": [float(x) for x in fuzzy["scores"]],
        "ga_history": [],
        "metrics": {
            "RMSE": rmse,
            "MAE": mae,
        },
        "latest_signal": fuzzy["labels"][-1] if fuzzy["labels"] else "Hold",
        "next_day_predicted_close": float(predicted[-1]),
        "next_trading_date": next_trading_date,
        "confidence": mean_fuzzy_conf,
        "processing_time_seconds": 0.0,
        "artifact_summary": {
            "best_ticker": summary.get("best_ticker"),
            "best_rmse": summary.get("best_rmse"),
            "average_rmse": summary.get("average_rmse"),
            "average_directional_accuracy": summary.get("average_directional_accuracy"),
        },
    }


def _build_response(
    ticker: str,
    model_result,
    prepared,
    params: Dict,
    ga_history: List[Dict],
    processing_time: float,
) -> Dict:
    """Compose frontend-ready response payload."""
    split = prepared.split
    frame = prepared.feature_frame

    # Align indicator arrays with the test segment used by the model.
    indicator_tail = frame.iloc[-len(split.y_test) :]
    rsi_values = indicator_tail["RSI"].astype(float).tolist()
    macd_values = indicator_tail["MACD_Signal"].astype(float).tolist()

    pred = model_result.predicted_prices
    actual = model_result.actual_prices

    # Fuzzy uses expected % change context together with RSI and MACD signal line.
    price_change_pct = (((pred - actual) / np.maximum(actual, 1e-8)) * 100.0).tolist()
    fuzzy = batch_signals(price_change_pct, rsi_values, macd_values)

    latest_conf = estimate_confidence(model_result.model, split.x_test[-1])
    mean_fuzzy_conf = float(np.mean(fuzzy["confidences"])) if fuzzy["confidences"] else 0.0
    confidence = float((latest_conf + mean_fuzzy_conf) / 2.0)

    # True next-session forecast from the latest available lookback window.
    x_next = np.expand_dims(prepared.scaled_features[-prepared.lookback :], axis=0)
    next_scaled = model_result.model.predict(x_next, verbose=0).flatten()
    next_price = float(inverse_close(prepared.scaler, next_scaled)[0])
    next_trading_date = str((pd.to_datetime(frame.index[-1]) + BDay(1)).date())

    response = {
        "ticker": ticker,
        "hyperparameters": params,
        "dates": split.test_dates,
        "actual_prices": [float(x) for x in actual.tolist()],
        "predicted_prices": [float(x) for x in pred.tolist()],
        "rsi": rsi_values,
        "macd": macd_values,
        "price_change_pct": [float(x) for x in price_change_pct],
        "signals": fuzzy["labels"],
        "signal_scores": [float(x) for x in fuzzy["scores"]],
        "ga_history": ga_history,
        "metrics": {
            "RMSE": float(model_result.rmse),
            "MAE": float(model_result.mae),
        },
        "latest_signal": fuzzy["labels"][-1] if fuzzy["labels"] else "Hold",
        "next_day_predicted_close": next_price,
        "next_trading_date": next_trading_date,
        "confidence": confidence,
        "processing_time_seconds": float(round(processing_time, 3)),
    }
    return response


def _run_pipeline(ticker: str, run_ga: bool) -> Dict:
    """Execute full or quick analysis path depending on run_ga flag."""
    start = time.perf_counter()

    ticker = ticker.upper().strip()
    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker cannot be empty.")

    params = DEFAULT_PARAMS.copy()
    ga_history: List[Dict] = []

    if run_ga:
        best_params, ga_history = optimize_hyperparameters(ticker=ticker, generations=15, population_size=10)
        params.update(best_params)

    prepared = prepare_data(ticker=ticker, lookback=int(params["lookback_window"]))
    model_result = train_and_evaluate(
        prepared,
        units=int(params["lstm_units"]),
        dropout=float(params["dropout_rate"]),
        learning_rate=float(params["learning_rate"]),
        epochs=35 if run_ga else 20,
        batch_size=32,
        verbose=0,
    )

    elapsed = time.perf_counter() - start
    return _build_response(
        ticker=ticker,
        model_result=model_result,
        prepared=prepared,
        params=params,
        ga_history=ga_history,
        processing_time=elapsed,
    )


def _predict_one_quick(ticker: str) -> Dict:
    start = time.perf_counter()
    artifact_response = _load_artifact_response(ticker)
    if artifact_response is not None:
        artifact_response["processing_time_seconds"] = float(round(time.perf_counter() - start, 3))
        return artifact_response
    return _run_pipeline(ticker=ticker, run_ga=False)


@app.get("/api/tickers")
def get_tickers() -> Dict:
    """Return a curated list of popular symbols for quick user selection."""
    started = time.perf_counter()
    artifact_tickers: List[str] = []
    pred_dir = ARTIFACTS_DIR / "predictions"
    if pred_dir.exists():
        for p in pred_dir.glob("*_predictions.csv"):
            key = p.stem.replace("_predictions", "")
            artifact_tickers.append(_file_key_to_ticker(key))

    merged = sorted(set(POPULAR_TICKERS + artifact_tickers))
    return {
        "tickers": merged,
        "processing_time_seconds": float(round(time.perf_counter() - started, 3)),
    }


@app.get("/api/predict")
def quick_predict(ticker: str = Query("AAPL", min_length=1)) -> Dict:
    """Quick path: no GA optimization, default LSTM settings only."""
    try:
        return _predict_one_quick(ticker=ticker)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Quick predict failed: {exc}") from exc


@app.get("/api/predict/batch")
def quick_predict_batch(tickers: str = Query(..., min_length=1)) -> Dict:
    """Batch quick predictions for up to 10 comma-separated tickers."""
    started = time.perf_counter()
    raw = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    unique: List[str] = []
    for t in raw:
        if t not in unique:
            unique.append(t)

    if not unique:
        raise HTTPException(status_code=400, detail="At least one ticker is required.")
    if len(unique) > 10:
        raise HTTPException(status_code=400, detail="Select at most 10 tickers.")

    results: List[Dict] = []
    errors: List[Dict] = []
    for t in unique:
        try:
            results.append(_predict_one_quick(ticker=t))
        except Exception as exc:
            errors.append({"ticker": t, "error": str(exc)})

    return {
        "requested": unique,
        "count": len(results),
        "results": results,
        "errors": errors,
        "processing_time_seconds": float(round(time.perf_counter() - started, 3)),
    }


@app.get("/api/analyze")
def analyze(
    ticker: str = Query("AAPL", min_length=1),
    run_ga: bool = Query(False),
    use_artifacts: bool = Query(True),
) -> Dict:
    """Full pipeline endpoint with optional GA optimization phase."""
    try:
        if use_artifacts and not run_ga:
            start = time.perf_counter()
            artifact_response = _load_artifact_response(ticker)
            if artifact_response is not None:
                artifact_response["processing_time_seconds"] = float(round(time.perf_counter() - start, 3))
                return artifact_response
        return _run_pipeline(ticker=ticker, run_ga=run_ga)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc
