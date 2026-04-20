"""LSTM model utilities for StockSense.

This module builds and trains a 2-layer LSTM for next-day close prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow import keras

from data_pipeline import inverse_close


@dataclass
class ModelResult:
    """Container with model and prediction artifacts."""

    model: keras.Model
    predicted_prices: np.ndarray
    actual_prices: np.ndarray
    rmse: float
    mae: float
    val_loss: float


def build_lstm_model(
    n_features: int,
    units: int = 64,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """Build a 2-layer LSTM architecture with configurable hyperparameters."""
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(None, n_features)),
            keras.layers.LSTM(units, return_sequences=True),
            keras.layers.Dropout(dropout),
            keras.layers.LSTM(max(16, units // 2), return_sequences=False),
            keras.layers.Dropout(dropout),
            keras.layers.Dense(1),
        ]
    )

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mse")
    return model


def train_and_evaluate(
    prepared,
    units: int = 64,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
    epochs: int = 30,
    batch_size: int = 32,
    verbose: int = 0,
) -> ModelResult:
    """Train model and compute RMSE/MAE on the holdout test split."""
    tf.keras.utils.set_random_seed(42)

    split = prepared.split
    model = build_lstm_model(
        n_features=split.x_train.shape[2],
        units=units,
        dropout=dropout,
        learning_rate=learning_rate,
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        split.x_train,
        split.y_train,
        validation_data=(split.x_val, split.y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        callbacks=callbacks,
    )

    y_pred_scaled = model.predict(split.x_test, verbose=0).flatten()
    y_true_scaled = split.y_test.flatten()

    y_pred = inverse_close(prepared.scaler, y_pred_scaled)
    y_true = inverse_close(prepared.scaler, y_true_scaled)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    val_loss = float(min(history.history.get("val_loss", [np.inf])))

    return ModelResult(
        model=model,
        predicted_prices=y_pred,
        actual_prices=y_true,
        rmse=rmse,
        mae=mae,
        val_loss=val_loss,
    )


def estimate_confidence(model: keras.Model, x_last: np.ndarray, passes: int = 20) -> float:
    """Estimate confidence by Monte Carlo dropout dispersion on the last sample."""
    if x_last.ndim == 2:
        x_last = np.expand_dims(x_last, axis=0)

    preds = []
    for _ in range(passes):
        pred = model(x_last, training=True).numpy().squeeze()
        preds.append(float(pred))

    std = float(np.std(preds))
    # Convert uncertainty to a bounded confidence percentage.
    confidence = max(0.0, min(100.0, (1.0 - std) * 100.0))
    return confidence
