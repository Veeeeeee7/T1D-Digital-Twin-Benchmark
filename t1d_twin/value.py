"""Reward and clinical glucose metrics"""
from __future__ import annotations

import numpy as np


_R_G = 3.550313
_R_A = 0.835334
_R_B = 3.793357

TIR_LOW = 70.0
TIR_HIGH = 180.0

# Smallest glucose passed to log(); guards against zeros / sensor floor.
_G_FLOOR = 1.0

# CGM measurement range.
CGM_MIN = 40.0
CGM_MAX = 400.0


def _as_array(glucose) -> np.ndarray:
    """Coerce input to a 1-D float array and validate it is non-empty."""
    g = np.asarray(glucose, dtype=float).ravel()
    if g.size == 0:
        raise ValueError("glucose series is empty")
    return g


def _risk_transform(glucose) -> np.ndarray:
    """f(g) = _R_G * (ln(g)**_R_A - _R_B), with g clamped to the CGM range."""
    g = np.clip(_as_array(glucose), max(CGM_MIN, _G_FLOOR), CGM_MAX)
    return _R_G * (np.log(g) ** _R_A - _R_B)


def magni_risk(glucose) -> np.ndarray:
    """Per-sample symmetrized glucose risk ``10 * f(g)**2`` [risk units]."""
    f = _risk_transform(glucose)
    return 10 * f * f


def risk_low_high(glucose) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample (low_risk, high_risk) split of :func:`magni_risk`."""
    f = _risk_transform(glucose)
    r = 10 * f * f
    low = np.where(f < 0.0, r, 0.0)
    high = np.where(f > 0.0, r, 0.0)
    return low, high


def lbgi(glucose) -> float:
    """Low Blood Glucose Index: mean of the hypoglycemic risk component."""
    low, _ = risk_low_high(glucose)
    return float(np.mean(low))


def hbgi(glucose) -> float:
    """High Blood Glucose Index: mean of the hyperglycemic risk component."""
    _, high = risk_low_high(glucose)
    return float(np.mean(high))


def mean_glucose(glucose) -> float:
    """Mean glucose [mg/dL]."""
    return float(np.mean(_as_array(glucose)))


def time_in_range(glucose, low: float = TIR_LOW, high: float = TIR_HIGH) -> float:
    """Percent of samples with ``low <= g <= high`` (default 70-180 mg/dL)."""
    g = _as_array(glucose)
    return float(100.0 * np.mean((g >= low) & (g <= high)))


def time_below_range(glucose, low: float = TIR_LOW) -> float:
    """Percent of samples with ``g < low`` (default <70 mg/dL)."""
    g = _as_array(glucose)
    return float(100.0 * np.mean(g < low))


def time_above_range(glucose, high: float = TIR_HIGH) -> float:
    """Percent of samples with ``g > high`` (default >180 mg/dL)."""
    g = _as_array(glucose)
    return float(100.0 * np.mean(g > high))


def reward(glucose, w_low: float = 1.0, w_high: float = 1.0) -> float:
    """Default scalar objective: ``-sum(magni_risk(glucose))``."""
    if w_low == 1.0 and w_high == 1.0:
        return float(-np.sum(magni_risk(glucose)))
    low, high = risk_low_high(glucose)
    return float(-np.sum(w_low * low + w_high * high))


def clinical_metrics(glucose) -> dict:
    """Keys: ``mean_glucose``, ``tir``, ``tbr``, ``tar`` (percent), ``lbgi``, ``hbgi``,
    ``magni_risk_sum``, ``magni_risk_mean``, ``reward``.
    """
    g = _as_array(glucose)
    low, high = risk_low_high(g)
    risk = low + high
    return {
        "mean_glucose": float(np.mean(g)),
        "tir": time_in_range(g),
        "tbr": time_below_range(g),
        "tar": time_above_range(g),
        "lbgi": float(np.mean(low)),
        "hbgi": float(np.mean(high)),
        "magni_risk_sum": float(np.sum(risk)),
        "magni_risk_mean": float(np.mean(risk)),
        "reward": float(-np.sum(risk)),
    }