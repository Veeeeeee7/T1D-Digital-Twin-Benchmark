"""Linear impulse-response twin — the benchmark's non-mechanistic baseline."""
from __future__ import annotations

from typing import Optional

import numpy as np

from .twin import Twin
from .simglucose_adapter import RunResult

#: Insulin memory, in CGM samples.
L_INS_DEFAULT: int = 48

#: Carb memory, in CGM samples.
L_CHO_DEFAULT: int = 36

#: Ridge penalty.
LAM_DEFAULT: float = 10.0

#: Training days consumed — **every shipped day**; see
#: ``run_benchmark.METHOD_FIT_DAYS`` for the resolution and the measurement.
FIT_DAYS_DEFAULT: int = 0        # 0 = all shipped days


def bin_to_sample_grid(x_dt: np.ndarray, reps: int) -> np.ndarray:
    """dt-grid rate -> CGM-grid mean rate. ``(T,) -> (T // reps,)``."""
    x = np.asarray(x_dt, dtype=float)
    n = (len(x) // reps) * reps
    return x[:n].reshape(-1, reps).mean(axis=1)


def design_matrix(ins: np.ndarray, cho: np.ndarray, Ib: float,
                  l_ins: int, l_cho: int) -> np.ndarray:
    """Causal lag matrix on the CGM grid, ``(n, l_ins + l_cho + 2)``."""
    ins = np.asarray(ins, dtype=float)
    cho = np.asarray(cho, dtype=float)
    n = len(ins)
    ip = np.concatenate([np.full(l_ins, float(Ib)), ins])
    cp = np.concatenate([np.zeros(l_cho), cho])
    xi = np.stack([ip[l_ins - l: l_ins - l + n] for l in range(l_ins + 1)], axis=1)
    xc = np.stack([cp[l_cho - l: l_cho - l + n] for l in range(l_cho + 1)], axis=1)
    return np.concatenate([xi, xc], axis=1)


def fit_ridge(X: np.ndarray, y: np.ndarray, lam: float) -> tuple[np.ndarray, float]:
    """Closed-form ridge with an UNPENALISED intercept; returns ``(w, b)``."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    xm, ym = X.mean(axis=0), float(y.mean())
    Xc = X - xm
    A = Xc.T @ Xc + float(lam) * np.eye(X.shape[1])
    w = np.linalg.solve(A, Xc.T @ (y - ym))
    return w, float(ym - xm @ w)


class LinearTwin(Twin):
    """Point-estimate twin over a fitted pair of impulse-response kernels."""

    def __init__(self, w: np.ndarray, b: float, Ib: float, sample_time: float,
                 dt: float = 1.0, l_ins: int = L_INS_DEFAULT,
                 l_cho: int = L_CHO_DEFAULT, sensor_name: str = "GuardianRT",
                 lam: float = LAM_DEFAULT, fit_rmse: float | None = None) -> None:
        self.w = np.asarray(w, dtype=float)
        self.b = float(b)
        self.Ib = float(Ib)
        self.sample_time = float(sample_time)
        self.dt = float(dt)
        self.l_ins = int(l_ins)
        self.l_cho = int(l_cho)
        self.sensor_name = sensor_name
        self.lam = float(lam)
        self.fit_rmse = fit_rmse
        exp = self.l_ins + self.l_cho + 2
        if self.w.shape != (exp,):
            raise ValueError(f"weight vector is {self.w.shape}, expected ({exp},) "
                             f"for l_ins={self.l_ins}, l_cho={self.l_cho}")

    # ------------------------------------------------------------------
    # Twin ABC
    # ------------------------------------------------------------------
    def predict_ig(self, insulin: np.ndarray, cho: np.ndarray,
                   n_samples: Optional[int] = None):
        """Predicted IG on the **dt grid**, as the ABC requires."""
        reps = int(round(self.sample_time / self.dt))
        ins5 = bin_to_sample_grid(insulin, reps)
        cho5 = bin_to_sample_grid(cho, reps)
        X = design_matrix(ins5, cho5, self.Ib, self.l_ins, self.l_cho)
        pred5 = X @ self.w + self.b
        ig = np.repeat(pred5, reps)
        if len(ig) < len(insulin):        # trailing partial bin, benchmark-impossible
            ig = np.concatenate([ig, np.full(len(insulin) - len(ig), ig[-1])])
        return ig, np.tile(ig, (2, 1))

    def summary(self) -> dict:
        """Kernel summaries. No physiological parameters exist to report."""
        wi = self.w[: self.l_ins + 1]
        wc = self.w[self.l_ins + 1:]
        out = {
            "intercept": self.b,
            "insulin_gain": float(wi.sum()),
            "insulin_peak_lag_min": float(np.argmax(np.abs(wi)) * self.sample_time),
            "cho_gain": float(wc.sum()),
            "cho_peak_lag_min": float(np.argmax(np.abs(wc)) * self.sample_time),
        }
        return {k: {"median": float(v), "ci95": (float(v), float(v))}
                for k, v in out.items()}


# ---------------------------------------------------------------------------
# Glue
# ---------------------------------------------------------------------------
def fit_from_arrays(cgm: np.ndarray, insulin: np.ndarray, cho: np.ndarray,
                    Ib: float, sample_time: float, dt: float = 1.0,
                    l_ins: int = L_INS_DEFAULT, l_cho: int = L_CHO_DEFAULT,
                    lam: float = LAM_DEFAULT,
                    sensor_name: str = "GuardianRT") -> LinearTwin:
    """Fit from stacked arrays. ``cgm`` is CGM-grid, the inputs are dt-grid."""
    reps = int(round(sample_time / dt))
    X = design_matrix(bin_to_sample_grid(insulin, reps),
                      bin_to_sample_grid(cho, reps), Ib, l_ins, l_cho)
    y = np.asarray(cgm, dtype=float)
    if len(X) != len(y):
        raise ValueError(f"{len(X)} covariate rows vs {len(y)} CGM samples")
    w, b = fit_ridge(X, y, lam)
    rmse = float(np.sqrt(np.mean((X @ w + b - y) ** 2)))
    return LinearTwin(w, b, Ib, sample_time, dt, l_ins, l_cho, sensor_name, lam,
                      fit_rmse=rmse)


def identify_twin_from_runs(runs, dt: float = 1.0,
                            l_ins: int = L_INS_DEFAULT,
                            l_cho: int = L_CHO_DEFAULT,
                            lam: float = LAM_DEFAULT,
                            sensor_name: str = "GuardianRT",
                            seed: int = 42) -> LinearTwin:
    """Identify a ``LinearTwin`` from one or more runs (each an identification day)."""
    runs = list(runs) if not isinstance(runs, RunResult) else [runs]
    if not runs:
        raise ValueError("need at least one run to identify a LinearTwin")

    Xs, ys = [], []
    Ib = None
    for run in runs:
        _, insulin, cho = run.replaybg_inputs(dt=dt)
        if Ib is None:
            Ib = float(insulin[0])
        reps = int(round(float(run.sample_time) / dt))
        Xs.append(design_matrix(bin_to_sample_grid(insulin, reps),
                                bin_to_sample_grid(cho, reps), Ib, l_ins, l_cho))
        ys.append(np.asarray(run.cgm(), dtype=float))
    X, y = np.vstack(Xs), np.concatenate(ys)
    if len(X) != len(y):
        raise ValueError(f"{len(X)} covariate rows vs {len(y)} CGM samples")
    w, b = fit_ridge(X, y, lam)
    rmse = float(np.sqrt(np.mean((X @ w + b - y) ** 2)))
    return LinearTwin(w, b, Ib, float(runs[0].sample_time), dt, l_ins, l_cho,
                      sensor_name, lam, fit_rmse=rmse)


def identify_twin_from_run(run: RunResult, dt: float = 1.0, **kw) -> LinearTwin:
    """Single-run convenience wrapper, matching the other methods' entry point."""
    return identify_twin_from_runs([run], dt=dt, **kw)
