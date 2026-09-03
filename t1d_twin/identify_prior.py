"""Prior (population) twin — the benchmark's NO-PERSONALIZATION baseline."""
from __future__ import annotations

import numpy as np

from . import replaybg_priors as _priors
from .identify_mcmc import DEFAULT_N_POSTERIOR, DEFAULT_SIGMA, MCMCTwin
from .replaybg_model import THETA_NAMES
from .simglucose_adapter import RunResult

#: Ensemble size.
N_DRAWS_DEFAULT: int = DEFAULT_N_POSTERIOR

#: Modes for :func:`draw_prior_theta`.
MODES = ("predictive", "center")


def draw_prior_theta(n: int = N_DRAWS_DEFAULT, seed: int = 42,
                     mode: str = "predictive") -> np.ndarray:
    """``(n, 8)`` natural-space theta drawn from the published ReplayBG prior."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    n = int(n)
    if n < 1:
        raise ValueError(f"need at least one draw, got {n}")

    if mode == "center":
        c = _priors.prior_center()
        theta = np.array([[c[name] for name in THETA_NAMES]], dtype=float)
        if not _in_support(theta).all():
            raise RuntimeError(
                "the published prior centre is outside the support box; the "
                "prior module and identify_mcmc have drifted apart")
        return np.repeat(theta, n, axis=0)

    rng = np.random.default_rng(seed)
    kept, need = [], n
    while need > 0:
        phi = _priors.sample_prior_phi(max(need * 2, 64), rng)
        theta = _priors.phi_to_theta(phi)
        theta = theta[_in_support(theta)]
        if len(theta):
            kept.append(theta)
            need = n - sum(len(k) for k in kept)
    return np.concatenate(kept, axis=0)[:n]


def _in_support(theta: np.ndarray) -> np.ndarray:
    """Boolean mask of rows inside the published support box."""
    theta = np.atleast_2d(np.asarray(theta, dtype=float))
    return np.all((theta >= _priors.SUPPORT_LO) & (theta <= _priors.SUPPORT_HI),
                  axis=1)


class PriorTwin(MCMCTwin):
    """A ReplayBG twin whose ensemble is the PRIOR, not a posterior."""

    def __init__(self, theta_prior: np.ndarray, Ib: float, sample_time: float,
                 dt: float = 1.0, sensor_name: str = "GuardianRT",
                 sigma: float = DEFAULT_SIGMA,
                 mode: str = "predictive") -> None:
        super().__init__(theta_post=theta_prior, Ib=Ib, sample_time=sample_time,
                         dt=dt, sensor_name=sensor_name, sigma=sigma)
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.mode = mode

    #: Alias that reads honestly at the call site.
    @property
    def theta_prior(self) -> np.ndarray:
        """The prior ensemble, aliased for symmetry with the posterior arms."""
        return self.theta_post

    def summary(self) -> dict:
        """Per-parameter PRIOR medians and 95% intervals."""
        out = super().summary()
        out["_meta"] = {"mode": self.mode, "personalized": False,
                        "n_draws": int(self.theta_post.shape[0])}
        return out


# ---------------------------------------------------------------------------
# Glue — signature-compatible with the other methods
# ---------------------------------------------------------------------------
def identify_twin_from_run(run: RunResult, dt: float = 1.0,
                           n_draws: int = N_DRAWS_DEFAULT,
                           sigma: float = DEFAULT_SIGMA,
                           sensor_name: str = "GuardianRT",
                           seed: int = 42,
                           mode: str = "predictive") -> PriorTwin:
    """Build a :class:`PriorTwin` from a run WITHOUT reading its CGM."""
    _, insulin, _ = run.replaybg_inputs(dt=dt)
    Ib = float(insulin[0])
    theta = draw_prior_theta(n=n_draws, seed=seed, mode=mode)
    return PriorTwin(theta_prior=theta, Ib=Ib, sample_time=float(run.sample_time),
                     dt=dt, sensor_name=sensor_name, sigma=sigma, mode=mode)


def identify_twin_from_runs(runs, dt: float = 1.0, **kw) -> PriorTwin:
    """Multi-run wrapper for signature uniformity."""
    runs = [runs] if isinstance(runs, RunResult) else list(runs)
    if not runs:
        raise ValueError("need at least one run to identify a PriorTwin")
    return identify_twin_from_run(runs[0], dt=dt, **kw)
