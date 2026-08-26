"""CGM sensor models for twin output generation."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
import pkg_resources

from simglucose.sensor.noise_gen import CGMNoise

_SENSOR_PARA_FILE = pkg_resources.resource_filename(
    "simglucose", "params/sensor_params.csv"
)


def _sensor_params(sensor_name: str, sample_time: float | None = None) -> pd.Series:
    """Load simglucose sensor params, optionally overriding ``sample_time``."""
    return _sensor_params_cached(sensor_name, sample_time).copy()


@lru_cache(maxsize=32)
def _sensor_params_cached(sensor_name: str, sample_time: float | None):
    """Cached CSV read. Callers get a copy so the cache stays clean."""
    df = pd.read_csv(_SENSOR_PARA_FILE)
    rows = df.loc[df.Name == sensor_name]
    if rows.empty:
        raise ValueError(
            f"unknown sensor {sensor_name!r}; available: {list(df.Name)}"
        )
    p = rows.squeeze().copy()
    if sample_time is not None:
        p["sample_time"] = float(sample_time)
    return p


def add_cgm_noise(
    ig_sampled: np.ndarray,
    seed: int = 1,
    sensor_name: str = "Dexcom",
    sample_time: float | None = None,
) -> np.ndarray:
    """Apply the UVA/Padova (Breton-Kovatchev) sensor model to sampled IG."""
    ig_sampled = np.asarray(ig_sampled, dtype=float)
    p = _sensor_params(sensor_name, sample_time)
    lo, hi = float(p["min"]), float(p["max"])

    noise_gen = CGMNoise(p, seed=seed)
    cgm = np.empty_like(ig_sampled)
    for i, ig_val in enumerate(ig_sampled):
        cgm[i] = float(np.clip(ig_val + next(noise_gen), lo, hi))
    return cgm


def gaussian_noise(
    ig_sampled: np.ndarray,
    sigma: float = 10.0,
    seed: int = 0,
) -> np.ndarray:
    """Add i.i.d. Gaussian noise to a sampled IG trace (white-noise fallback)."""
    ig_sampled = np.asarray(ig_sampled, dtype=float)
    rng = np.random.default_rng(seed)
    return ig_sampled + rng.normal(0.0, sigma, size=ig_sampled.shape)