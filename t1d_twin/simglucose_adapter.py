"""simglucose adapter."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class RunResult:
    """Standardized output of one simulation run."""
    df: pd.DataFrame
    BW: float
    sample_time: float
    patient_name: str
    sensor_name: str
    sensor_seed: int
    meta: dict = field(default_factory=dict)

    # --- convenience accessors -------------------------------------------------
    def cgm(self) -> np.ndarray:
        """Noisy CGM at sample times [mg/dL]."""
        return self.df["CGM"].to_numpy()

    def bg(self) -> np.ndarray:
        """Noise-free interstitial glucose (IG) at sample times [mg/dL]."""
        return self.df["BG"].to_numpy()

    def replaybg_inputs(self, dt: float = 1.0):
        """Inputs for the ReplayBG forward model on a regular ``dt``-minute grid."""
        reps = self.sample_time / dt
        if abs(reps - round(reps)) > 1e-9:
            raise ValueError(
                f"sample_time ({self.sample_time}) must be an integer multiple of dt ({dt})")
        reps = int(round(reps))
        insulin = np.repeat(self.df["insulin_mU_kg_min"].to_numpy(), reps)
        cho = np.repeat(self.df["CHO_mg_kg_min"].to_numpy(), reps)
        t_grid = np.arange(len(insulin)) * dt
        return t_grid, insulin, cho
