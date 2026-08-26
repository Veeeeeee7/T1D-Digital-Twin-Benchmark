"""Unscented Kalman Filter twin — joint state-parameter identification."""
from __future__ import annotations

import numpy as np

from . import replaybg_priors as _priors
from .identify_mcmc import DEFAULT_SIGMA, MCMCTwin
from .replaybg_model import (
    KE, VI, _deriv, _delayed_insulin, steady_state, theta_to_array,
)
from .simglucose_adapter import RunResult

N_STATES = 9
N_PHI = 8
N_AUG = N_STATES + N_PHI                     # 17

#: Scaled unscented transform parameters (Julier & Uhlmann; van der Merwe).
ALPHA: float = 1.0
BETA: float = 2.0                            # 2 is optimal for Gaussian priors
KAPPA: float = 0.0

#: Measurement noise std [mg/dL].
SIGMA_CGM: float = DEFAULT_SIGMA

#: Parameter random-walk noise, as a fraction of the PRIOR variance per CGM step.
Q_PHI_FRAC: float = 1e-4

#: Process noise on the physiological states, as a fraction of the steady-state
#: magnitude per CGM step.
Q_STATE_FRAC: float = 1e-6

#: Draws taken from the final Gaussian over ``phi`` to form the replay ensemble.
N_ENSEMBLE: int = 1000

#: Jitter added to a covariance whose Cholesky factorisation fails, escalating until
#: it succeeds.
_JITTER0: float = 1e-9


def sigma_weights(L: int = N_AUG, alpha: float = ALPHA, beta: float = BETA,
                  kappa: float = KAPPA) -> tuple[np.ndarray, np.ndarray, float]:
    """``(Wm, Wc, lambda_)`` for the scaled unscented transform."""
    lam = alpha ** 2 * (L + kappa) - L
    Wm = np.full(2 * L + 1, 1.0 / (2.0 * (L + lam)))
    Wc = Wm.copy()
    Wm[0] = lam / (L + lam)
    Wc[0] = lam / (L + lam) + (1.0 - alpha ** 2 + beta)
    return Wm, Wc, lam


def _chol_psd(P: np.ndarray) -> np.ndarray:
    """Lower Cholesky factor, with escalating jitter on failure."""
    P = 0.5 * (P + P.T)                      # symmetrise; kills drift from roundoff
    jitter = _JITTER0 * max(1.0, float(np.trace(P)) / P.shape[0])
    for _ in range(8):
        try:
            return np.linalg.cholesky(P + jitter * np.eye(P.shape[0]))
        except np.linalg.LinAlgError:
            jitter *= 100.0
    # Last resort: clip the spectrum.
    w, V = np.linalg.eigh(P)
    return V @ np.diag(np.sqrt(np.maximum(w, 1e-12)))


def _sigma_points(mu: np.ndarray, P: np.ndarray, lam: float) -> np.ndarray:
    """``(2L+1, L)`` sigma points about ``mu`` with spread set by ``P``."""
    L = mu.shape[0]
    S = _chol_psd((L + lam) * P)
    pts = np.empty((2 * L + 1, L))
    pts[0] = mu
    pts[1:L + 1] = mu + S.T
    pts[L + 1:] = mu - S.T
    return pts


def _propagate(x: np.ndarray, phi: np.ndarray, I_del: np.ndarray,
               cho: np.ndarray, Ipb: float, dt: float) -> np.ndarray:
    """RK4 the states of every sigma point over one CGM interval."""
    theta = _priors.phi_to_theta(phi)
    for m in range(len(I_del)):
        Id, ch = float(I_del[m]), float(cho[m])
        k1 = _deriv(x, Id, ch, theta, Ipb)
        k2 = _deriv(x + 0.5 * dt * k1, Id, ch, theta, Ipb)
        k3 = _deriv(x + 0.5 * dt * k2, Id, ch, theta, Ipb)
        k4 = _deriv(x + dt * k3, Id, ch, theta, Ipb)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


def _phi_bounds() -> tuple[np.ndarray, np.ndarray]:
    """Support box in phi space, from the published theta-space support."""
    return (_priors.theta_to_phi(_priors.SUPPORT_LO),
            _priors.theta_to_phi(_priors.SUPPORT_HI))


def run_ukf(cgm: np.ndarray, insulin: np.ndarray, cho: np.ndarray, Ib: float,
            sample_time: float = 5.0, dt: float = 1.0,
            sigma: float = SIGMA_CGM, seed: int = 42,
            q_phi_frac: float = Q_PHI_FRAC,
            q_state_frac: float = Q_STATE_FRAC,
            return_states: bool = False) -> dict:
    """One forward joint state-parameter UKF pass over the window."""
    cgm = np.asarray(cgm, dtype=float)
    insulin = np.asarray(insulin, dtype=float)
    cho = np.asarray(cho, dtype=float)
    reps = int(round(sample_time / dt))
    n = len(cgm)
    if len(insulin) < n * reps:
        raise ValueError(f"{len(insulin)} dt samples for {n} CGM samples at "
                         f"{reps} per interval; need {n * reps}")

    rng = np.random.default_rng(seed)
    # Prior mean and covariance in phi space, taken empirically from the SAME
    # published prior every other mechanistic arm uses rather than hand-set.
    phi_draws = _priors.sample_prior_phi(4096, rng)
    phi0 = phi_draws.mean(axis=0)
    P_phi0 = np.cov(phi_draws, rowvar=False)
    lo, hi = _phi_bounds()

    theta0 = _priors.phi_to_theta(phi0)
    x0 = np.atleast_2d(steady_state(theta0, Ib))[0]

    mu = np.concatenate([x0, phi0])
    P = np.zeros((N_AUG, N_AUG))
    # State block: start tight — the window opens at steady state, which is the same
    # assumption every arm's replay makes.
    P[:N_STATES, :N_STATES] = np.diag(np.maximum(np.abs(x0) * 1e-3, 1e-8) ** 2)
    P[N_STATES:, N_STATES:] = P_phi0

    Q = np.zeros((N_AUG, N_AUG))
    Q[:N_STATES, :N_STATES] = np.diag((np.maximum(np.abs(x0), 1e-6)
                                       * q_state_frac) ** 2)
    Q[N_STATES:, N_STATES:] = P_phi0 * q_phi_frac
    R = float(sigma) ** 2

    Wm, Wc, lam = sigma_weights(N_AUG)
    Ipb = Ib / (VI * KE)
    I_del = _delayed_insulin(insulin, Ib, dt)

    innov, n_jitter = [], 0
    x_hist = np.empty((n, N_STATES)) if return_states else None
    for k in range(n):
        sl = slice(k * reps, (k + 1) * reps)

        # --- predict -----------------------------------------------------
        pts = _sigma_points(mu, P, lam)
        # The previous order advanced the states using the RAW sigma-point parameters
        # and then stored the CLIPPED ones in Z.
        phis = np.clip(pts[:, N_STATES:], lo, hi)
        xs = _propagate(pts[:, :N_STATES].copy(), phis,
                        I_del[sl], cho[sl], Ipb, dt)
        Z = np.concatenate([xs, phis], axis=1)        # parameters are static
        if not np.all(np.isfinite(Z)):
            # A sigma point left the physical domain.
            n_jitter += 1
            continue

        mu = Wm @ Z
        d = Z - mu
        P = (d.T * Wc) @ d + Q

        # --- update ------------------------------------------------------
        pts = _sigma_points(mu, P, lam)
        y = pts[:, 8]                                  # IG is state index 8
        y_hat = float(Wm @ y)
        dy = y - y_hat
        S = float((Wc * dy) @ dy) + R
        Pxy = (pts - mu).T @ (Wc * dy)
        if not np.isfinite(S) or S <= 0:
            n_jitter += 1
            continue
        K = Pxy / S
        r = float(cgm[k]) - y_hat
        mu = mu + K * r
        P = P - np.outer(K, K) * S
        mu[N_STATES:] = np.clip(mu[N_STATES:], lo, hi)
        innov.append(r)
        if x_hist is not None:
            x_hist[k] = mu[:N_STATES]

    phi_mean = mu[N_STATES:]
    P_phi = P[N_STATES:, N_STATES:]
    out = {
        "phi_mean": phi_mean,
        "P_phi": 0.5 * (P_phi + P_phi.T),
        "x_final": mu[:N_STATES],
        "innov_rmse": float(np.sqrt(np.mean(np.square(innov)))) if innov else float("nan"),
        "n_skipped": int(n_jitter),
        "n_steps": int(n),
    }
    if x_hist is not None:
        out["x_hist"] = x_hist
    return out


def ensemble_from_gaussian(phi_mean: np.ndarray, P_phi: np.ndarray,
                           n: int = N_ENSEMBLE, seed: int = 42) -> np.ndarray:
    """``(n, 8)`` natural-space theta drawn from the filter's Gaussian over phi."""
    rng = np.random.default_rng(seed)
    lo, hi = _phi_bounds()
    L = _chol_psd(P_phi)
    phi = phi_mean + rng.standard_normal((int(n), N_PHI)) @ L.T
    phi = np.clip(phi, lo, hi)
    theta = _priors.phi_to_theta(phi)

    # If the mean ITSELF violated the ordering, every fallback violated it too and
    # nothing noticed.
    fallback = _project_ordering(np.clip(phi_mean, lo, hi), lo, hi)
    ok = (theta[:, 0] < theta[:, 1]) & (theta[:, 3] < theta[:, 2])
    if not ok.all():
        theta[~ok] = _priors.phi_to_theta(fallback)

    ok = (theta[:, 0] < theta[:, 1]) & (theta[:, 3] < theta[:, 2])
    if not ok.all():                      # cannot happen; assert rather than hope
        raise RuntimeError(
            f"{int((~ok).sum())}/{len(theta)} ensemble draws violate the ReplayBG "
            f"ordering constraints after projection — the fallback is broken")
    return theta


def _project_ordering(phi: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Nudge ``phi`` into ``ka2 < kd`` and ``kabs < kempt``, staying in the box."""
    phi = np.array(phi, dtype=float, copy=True)
    eps = 1e-3
    if phi[0] >= phi[1]:                                   # ka2 < kd
        phi[0] = max(lo[0], min(phi[1] - eps, hi[0]))
    if phi[3] >= phi[2]:                                   # kabs < kempt
        phi[3] = max(lo[3], min(phi[2] - eps, hi[3]))
    return phi


class UKFTwin(MCMCTwin):
    """A ReplayBG twin whose ensemble comes from a UKF, not a sampler."""

    def __init__(self, theta_ens: np.ndarray, Ib: float, sample_time: float,
                 dt: float = 1.0, sensor_name: str = "GuardianRT",
                 sigma: float = SIGMA_CGM, diagnostics: dict | None = None) -> None:
        super().__init__(theta_post=theta_ens, Ib=Ib, sample_time=sample_time,
                         dt=dt, sensor_name=sensor_name, sigma=sigma)
        self.diagnostics = dict(diagnostics or {})

    def summary(self) -> dict:
        """Per-parameter medians and intervals, plus the filter's own diagnostics."""
        out = super().summary()
        out["_meta"] = {
            "method": "ukf",
            "personalized": True,
            "innov_rmse": self.diagnostics.get("innov_rmse"),
            "n_skipped": self.diagnostics.get("n_skipped"),
            "n_steps": self.diagnostics.get("n_steps"),
        }
        return out


# ---------------------------------------------------------------------------
# Glue — signature-compatible with the other methods
# ---------------------------------------------------------------------------
def identify_twin_from_run(run: RunResult, dt: float = 1.0,
                           sigma: float = SIGMA_CGM,
                           sensor_name: str = "GuardianRT",
                           n_ensemble: int = N_ENSEMBLE,
                           seed: int = 42) -> UKFTwin:
    """Identify a :class:`UKFTwin` from one identification window."""
    cgm = run.cgm()
    _, insulin, cho = run.replaybg_inputs(dt=dt)
    Ib = float(insulin[0])

    res = run_ukf(cgm=cgm, insulin=insulin, cho=cho, Ib=Ib,
                  sample_time=float(run.sample_time), dt=dt, sigma=sigma,
                  seed=seed)
    theta_ens = ensemble_from_gaussian(res["phi_mean"], res["P_phi"],
                                       n=n_ensemble, seed=seed)
    return UKFTwin(theta_ens=theta_ens, Ib=Ib,
                   sample_time=float(run.sample_time), dt=dt,
                   sensor_name=sensor_name, sigma=sigma, diagnostics=res)


def identify_twin_from_runs(runs, dt: float = 1.0, **kw) -> UKFTwin:
    """Multi-run wrapper for signature uniformity; uses the first window."""
    runs = [runs] if isinstance(runs, RunResult) else list(runs)
    if not runs:
        raise ValueError("need at least one run to identify a UKFTwin")
    return identify_twin_from_run(runs[0], dt=dt, **kw)
