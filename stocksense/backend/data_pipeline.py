"""Data pipeline for StockSense.

This module handles market data collection, technical indicator engineering,
normalization, and LSTM-ready sliding-window sequence creation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


FEATURE_COLUMNS: List[str] = [
    "Close",
    "Volume",
    "RSI",
    "MACD",
    "MACD_Signal",
    "EMA20",
    "BB_Upper",
    "BB_Lower",
]


@dataclass
class SequenceSplit:
    """Container for time-series train/val/test splits."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    test_dates: List[str]


@dataclass
class PreparedData:
    """Container with all processed artifacts needed by model/GA/API layers."""

    ticker: str
    feature_frame: pd.DataFrame
    scaled_features: np.ndarray
    scaler: MinMaxScaler
    split: SequenceSplit
    lookback: int


def fetch_ohlcv(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """Fetch daily OHLCV data from Yahoo Finance."""
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError(f"No market data found for ticker '{ticker}'.")
    # Flatten possible multi-index columns returned by yfinance in some environments.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add RSI, MACD, EMA20, and Bollinger Bands to the frame."""
    out = df.copy()

    # RSI(14)
    delta = out["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["RSI"] = 100 - (100 / (1 + rs))

    # MACD and Signal
    ema_fast = out["Close"].ewm(span=12, adjust=False).mean()
    ema_slow = out["Close"].ewm(span=26, adjust=False).mean()
    out["MACD"] = ema_fast - ema_slow
    out["MACD_Signal"] = out["MACD"].ewm(span=9, adjust=False).mean()

    # EMA20
    out["EMA20"] = out["Close"].ewm(span=20, adjust=False).mean()

    # Bollinger Bands(20, 2 std)
    ma20 = out["Close"].rolling(20).mean()
    std20 = out["Close"].rolling(20).std()
    out["BB_Upper"] = ma20 + 2 * std20
    out["BB_Lower"] = ma20 - 2 * std20

    out = out.dropna().copy()
    if out.empty:
        raise ValueError("Indicator engineering produced an empty dataset.")
    return out


def normalize_features(df: pd.DataFrame) -> Tuple[np.ndarray, MinMaxScaler]:
    """Normalize selected features for neural sequence learning."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(df[FEATURE_COLUMNS])
    return scaled, scaler


def build_sequences(
    scaled_features: np.ndarray,
    dates: pd.Index,
    lookback: int = 60,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> SequenceSplit:
    """Create sliding windows and preserve temporal ordering in splits."""
    x_data: List[np.ndarray] = []
    y_data: List[float] = []
    sample_dates: List[str] = []

    for i in range(lookback, len(scaled_features)):
        x_data.append(scaled_features[i - lookback : i])
        y_data.append(scaled_features[i, 0])
        sample_dates.append(str(pd.to_datetime(dates[i]).date()))

    if len(x_data) < 100:
        raise ValueError("Not enough sequence samples; try lowering lookback.")

    x = np.array(x_data)
    y = np.array(y_data)

    train_end = int(len(x) * train_ratio)
    val_end = int(len(x) * (train_ratio + val_ratio))

    x_train, y_train = x[:train_end], y[:train_end]
    x_val, y_val = x[train_end:val_end], y[train_end:val_end]
    x_test, y_test = x[val_end:], y[val_end:]
    test_dates = sample_dates[val_end:]

    return SequenceSplit(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        test_dates=test_dates,
    )


def prepare_data(ticker: str, lookback: int = 60, period: str = "5y") -> PreparedData:
    """Run the full pipeline and return model-ready artifacts."""
    raw = fetch_ohlcv(ticker=ticker, period=period, interval="1d")
    feature_frame = compute_indicators(raw)
    scaled_features, scaler = normalize_features(feature_frame)
    split = build_sequences(scaled_features, feature_frame.index, lookback=lookback)

    return PreparedData(
        ticker=ticker,
        feature_frame=feature_frame,
        scaled_features=scaled_features,
        scaler=scaler,
        split=split,
        lookback=lookback,
    )


def inverse_close(scaler: MinMaxScaler, close_scaled: np.ndarray) -> np.ndarray:
    """Inverse-transform only close price from normalized values."""
    arr = np.array(close_scaled).reshape(-1, 1)
    if arr.size == 0:
        return np.array([])

    # Build a dummy feature matrix so MinMaxScaler can invert the first feature (Close).
    dummy = np.zeros((len(arr), len(FEATURE_COLUMNS)))
    dummy[:, 0] = arr[:, 0]
    inv = scaler.inverse_transform(dummy)
    return inv[:, 0]
