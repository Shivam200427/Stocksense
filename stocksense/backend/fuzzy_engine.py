"""Fuzzy inference engine for trading signal generation.

Mamdani FIS maps market context to a 0-100 score and then to a discrete action.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class FuzzySignalEngine:
    """Reusable fuzzy control system for stock signal scoring."""

    def __init__(self):
        # Antecedents (inputs) from project requirements.
        self.price_change_pct = ctrl.Antecedent(np.arange(-10, 10.1, 0.1), "price_change_pct")
        self.rsi = ctrl.Antecedent(np.arange(0, 100.1, 1), "rsi")
        self.macd_signal = ctrl.Antecedent(np.arange(-2, 2.01, 0.01), "macd_signal")

        # Consequent (output) score mapped to 5 classes.
        self.signal_score = ctrl.Consequent(np.arange(0, 100.1, 1), "signal_score")

        self._build_membership_functions()
        self._build_rules()

        self.control_system = ctrl.ControlSystem(self.rules)

    def _build_membership_functions(self):
        """Define triangular low/medium/high sets for each input."""
        self.price_change_pct["low"] = fuzz.trimf(self.price_change_pct.universe, [-10, -10, 0])
        self.price_change_pct["medium"] = fuzz.trimf(self.price_change_pct.universe, [-3, 0, 3])
        self.price_change_pct["high"] = fuzz.trimf(self.price_change_pct.universe, [0, 10, 10])

        self.rsi["low"] = fuzz.trimf(self.rsi.universe, [0, 0, 40])
        self.rsi["medium"] = fuzz.trimf(self.rsi.universe, [30, 50, 70])
        self.rsi["high"] = fuzz.trimf(self.rsi.universe, [60, 100, 100])

        self.macd_signal["low"] = fuzz.trimf(self.macd_signal.universe, [-2, -2, 0])
        self.macd_signal["medium"] = fuzz.trimf(self.macd_signal.universe, [-0.5, 0, 0.5])
        self.macd_signal["high"] = fuzz.trimf(self.macd_signal.universe, [0, 2, 2])

        self.signal_score["strong_sell"] = fuzz.trimf(self.signal_score.universe, [0, 0, 20])
        self.signal_score["sell"] = fuzz.trimf(self.signal_score.universe, [15, 30, 45])
        self.signal_score["hold"] = fuzz.trimf(self.signal_score.universe, [40, 50, 60])
        self.signal_score["buy"] = fuzz.trimf(self.signal_score.universe, [55, 70, 85])
        self.signal_score["strong_buy"] = fuzz.trimf(self.signal_score.universe, [80, 100, 100])

    def _build_rules(self):
        """Define 9 core Mamdani rules mixing momentum and oscillator context."""
        self.rules = [
            ctrl.Rule(self.price_change_pct["high"] & self.rsi["low"] & self.macd_signal["high"], self.signal_score["strong_buy"]),
            ctrl.Rule(self.price_change_pct["high"] & self.rsi["medium"] & self.macd_signal["high"], self.signal_score["buy"]),
            ctrl.Rule(self.price_change_pct["high"] & self.rsi["high"] & self.macd_signal["medium"], self.signal_score["hold"]),
            ctrl.Rule(self.price_change_pct["medium"] & self.rsi["low"] & self.macd_signal["high"], self.signal_score["buy"]),
            ctrl.Rule(self.price_change_pct["medium"] & self.rsi["medium"] & self.macd_signal["medium"], self.signal_score["hold"]),
            ctrl.Rule(self.price_change_pct["medium"] & self.rsi["high"] & self.macd_signal["low"], self.signal_score["sell"]),
            ctrl.Rule(self.price_change_pct["low"] & self.rsi["high"] & self.macd_signal["low"], self.signal_score["strong_sell"]),
            ctrl.Rule(self.price_change_pct["low"] & self.rsi["medium"] & self.macd_signal["low"], self.signal_score["sell"]),
            ctrl.Rule(self.price_change_pct["low"] & self.rsi["low"] & self.macd_signal["medium"], self.signal_score["hold"]),
        ]

    @staticmethod
    def map_score_to_label(score: float) -> str:
        """Map fuzzy score to categorical trading label."""
        if score >= 80:
            return "Strong Buy"
        if score >= 60:
            return "Buy"
        if score >= 40:
            return "Hold"
        if score >= 20:
            return "Sell"
        return "Strong Sell"

    def infer(self, price_change_pct: float, rsi: float, macd_signal: float) -> Dict:
        """Run one fuzzy inference step and return score, label, confidence."""
        sim = ctrl.ControlSystemSimulation(self.control_system)
        sim.input["price_change_pct"] = float(np.clip(price_change_pct, -10, 10))
        sim.input["rsi"] = float(np.clip(rsi, 0, 100))
        sim.input["macd_signal"] = float(np.clip(macd_signal, -2, 2))
        sim.compute()

        score = float(sim.output["signal_score"])
        label = self.map_score_to_label(score)

        # Confidence is higher the farther score is from the neutral hold midpoint.
        confidence = float(min(100.0, abs(score - 50.0) * 2.0))

        return {
            "score": score,
            "label": label,
            "confidence": confidence,
        }


def batch_signals(price_changes: List[float], rsi_values: List[float], macd_values: List[float]) -> Dict:
    """Vector-style helper to infer signals over aligned series."""
    if not (len(price_changes) == len(rsi_values) == len(macd_values)):
        raise ValueError("Fuzzy inputs must be aligned arrays with equal length.")

    engine = FuzzySignalEngine()
    out_scores: List[float] = []
    out_labels: List[str] = []
    out_conf: List[float] = []

    for pchg, rsi, macd in zip(price_changes, rsi_values, macd_values):
        result = engine.infer(pchg, rsi, macd)
        out_scores.append(result["score"])
        out_labels.append(result["label"])
        out_conf.append(result["confidence"])

    return {
        "scores": out_scores,
        "labels": out_labels,
        "confidences": out_conf,
    }
