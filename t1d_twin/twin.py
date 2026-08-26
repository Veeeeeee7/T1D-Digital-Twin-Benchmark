"""Common ``Twin`` interface for all twinning methods."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from .replaybg_model import sample_indices
from .sensor import add_cgm_noise, gaussian_noise
from .simglucose_adapter import RunResult


class Twin(ABC):
    """Abstract base class for a personalized glucose-dynamics digital twin."""

    # Subclasses set these in __init__; listed here for documentation.
    Ib: float
    sample_time: float
    dt: float
    sensor_name: str

    # ------------------------------------------------------------------
    # Abstract interface — subclasses must implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def predict_ig(
        self,
        insulin: np.ndarray,
        cho: np.ndarray,
        n_samples: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate IG under a new therapy."""

    @abstractmethod
    def summary(self) -> dict:
        """Human-readable parameter summary."""

    # ------------------------------------------------------------------
    # Concrete methods — provided by the base class
    # ------------------------------------------------------------------

    def generate_cgm(
        self,
        insulin: np.ndarray,
        cho: np.ndarray,
        seed: int = 1,
        use_sensor_model: bool = True,
        sigma_fallback: float = 10.0,
        n_samples: Optional[int] = None,
        add_noise: bool = True,
    ) -> np.ndarray:
        """Simulate CGM from the twin under a new therapy."""
        T = len(insulin)
        n_cgm = T // int(round(self.sample_time / self.dt))
        obs_idx = sample_indices(n_cgm, self.sample_time, self.dt)

        if n_samples is None:
            ig_median, _ = self.predict_ig(insulin, cho, n_samples=None)
            ig_obs = ig_median[obs_idx]
            if not add_noise:
                return ig_obs                      # noise-free IG at CGM times
            if use_sensor_model:
                return add_cgm_noise(ig_obs, seed=seed,
                                     sensor_name=self.sensor_name,
                                     sample_time=self.sample_time)
            else:
                return gaussian_noise(ig_obs, sigma=sigma_fallback, seed=seed)
        else:
            # Return an ensemble: one noisy trace per posterior sample.
            ig_ens = self._ig_ensemble(insulin, cho, n_samples)  # (n_samples, T)
            ig_obs_ens = ig_ens[:, obs_idx]                     # (n_samples, n_cgm)
            cgm_ens = np.empty_like(ig_obs_ens)
            for i in range(n_samples):
                if use_sensor_model:
                    cgm_ens[i] = add_cgm_noise(
                        ig_obs_ens[i], seed=seed + i,
                        sensor_name=self.sensor_name,
                        sample_time=self.sample_time,
                    )
                else:
                    cgm_ens[i] = gaussian_noise(
                        ig_obs_ens[i], sigma=sigma_fallback, seed=seed + i,
                    )
            return cgm_ens

    def _ig_ensemble(
        self,
        insulin: np.ndarray,
        cho: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        """Return a (n_samples, T) IG ensemble."""
        ig_median, _ = self.predict_ig(insulin, cho, n_samples=n_samples)
        return np.tile(ig_median, (n_samples, 1))

    def generate_cgm_band(
        self,
        insulin: np.ndarray,
        cho: np.ndarray,
        n_samples: int = 200,
        seed: int = 1,
        use_sensor_model: bool = True,
        sigma_fallback: float = 10.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Standardized CGM-scale prediction: median + 25-75%% band."""
        cgm_ens = self.generate_cgm(
            insulin, cho,
            seed=seed,
            use_sensor_model=use_sensor_model,
            sigma_fallback=sigma_fallback,
            n_samples=n_samples,
        )                                            # (n_samples, n_cgm)
        cgm_median = np.median(cgm_ens, axis=0)
        cgm_iqr = np.percentile(cgm_ens, [25.0, 75.0], axis=0)

        n_cgm = cgm_ens.shape[1]
        t_cgm_min = (np.arange(n_cgm) + 1) * self.sample_time
        return t_cgm_min, cgm_median, cgm_iqr

    def replay_run(
        self,
        run: RunResult,
        seed: int = 1,
        use_sensor_model: bool = True,
        sigma_fallback: float = 10.0,
        add_noise: bool = True,
    ) -> np.ndarray:
        """Predict CGM (or noise-free IG) for the therapy recorded in ``run``."""
        _, insulin, cho = run.replaybg_inputs(dt=self.dt)
        return self.generate_cgm(
            insulin, cho,
            seed=seed,
            use_sensor_model=use_sensor_model,
            sigma_fallback=sigma_fallback,
            add_noise=add_noise,
        )

    def replay_run_band(
        self,
        run: RunResult,
        n_samples: int = 200,
        seed: int = 1,
        use_sensor_model: bool = True,
        sigma_fallback: float = 10.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``generate_cgm_band`` for the therapy recorded in ``run``."""
        _, insulin, cho = run.replaybg_inputs(dt=self.dt)
        return self.generate_cgm_band(
            insulin, cho,
            n_samples=n_samples,
            seed=seed,
            use_sensor_model=use_sensor_model,
            sigma_fallback=sigma_fallback,
        )

    def __repr__(self) -> str:
        s = self.summary()
        cls = type(self).__name__
        lines = [f"{cls} summary:"]
        for name, v in s.items():
            lo, hi = v["ci95"]
            lines.append(
                f"  {name:6s}: median={v['median']:.4g}  "
                f"95%CI=[{lo:.4g}, {hi:.4g}]"
            )
        return "\n".join(lines)