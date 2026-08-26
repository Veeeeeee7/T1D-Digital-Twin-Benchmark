"""Bayesian MCMC identification of a ReplayBG digital twin (Cappon et al., 2023)."""
from __future__ import annotations

import os
import warnings
from typing import Optional

import numpy as np

# Show a progress log only when the benchmark runner is in verbose mode
# (experiments.logsetup exports T1D_VERBOSE).
_PROGRESS_DEFAULT = os.environ.get("T1D_VERBOSE") == "1"

from .replaybg_model import (
    THETA_NAMES,
    theta_to_array, simulate, sample_indices,
)
from . import replaybg_priors as _priors
from .copula import fit_t_copula
from .simglucose_adapter import RunResult
from .twin import Twin

# ---------------------------------------------------------------------------
# Prior bounds in natural (un-transformed) space
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Support envelope (natural space).
# ---------------------------------------------------------------------------
PRIOR_LO = _priors.SUPPORT_LO.copy()
PRIOR_HI = _priors.SUPPORT_HI.copy()

# Keep old private names as aliases for test-file compatibility
_PRIOR_LO = PRIOR_LO
_PRIOR_HI = PRIOR_HI

# Default noise std [mg/dL] — matches Dexcom empirical std (~10 mg/dL)
DEFAULT_SIGMA: float = 10.0

# Default MCMC hyperparameters — ReplayBG single-chain budget (Hoang et al.
DEFAULT_BURN: int = 10_000
DEFAULT_NSAMPLE: int = 5_000
DEFAULT_N_POSTERIOR: int = 1000

# Adaptive Metropolis-within-Gibbs proposal defaults.
TARGET_ACCEPT: float = 0.44          # optimal single-component acceptance rate
ADAPT_BATCH: int = 50                # iterations per adaptation update
# Per-component initial random-walk std in phi-space.
_INIT_STEP = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 2.0])


# ---------------------------------------------------------------------------
# Parameter transforms phi <-> theta
# ---------------------------------------------------------------------------

_theta_to_phi = _priors.theta_to_phi
_phi_to_theta = _priors.phi_to_theta


# ---------------------------------------------------------------------------
# Log-prior (vectorised over batch B)
# ---------------------------------------------------------------------------

def log_prior(phi_batch: np.ndarray) -> np.ndarray:
    """Informative published (Cappon et al.) log-prior in phi-space; (B,8)->(B,)."""
    return _priors.log_prior_phi(np.atleast_2d(phi_batch))


# ---------------------------------------------------------------------------
# Log-likelihood (vectorised, batched simulate)
# ---------------------------------------------------------------------------

def log_likelihood(
    phi_batch: np.ndarray,
    cgm: np.ndarray,
    insulin: np.ndarray,
    cho: np.ndarray,
    Ib: float,
    sample_time: float,
    dt: float = 1.0,
    sigma: float = DEFAULT_SIGMA,
) -> np.ndarray:
    """Gaussian CGM log-likelihood for a batch of phi vectors."""
    theta_b = _phi_to_theta(np.atleast_2d(phi_batch))
    n = len(cgm)
    obs_idx = sample_indices(n, sample_time, dt)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _, ig_b = simulate(theta_b, insulin, cho, Ib=Ib, dt=dt)   # (B, T)

    ig_obs = ig_b[:, obs_idx]                                      # (B, n)
    finite_mask = np.all(np.isfinite(ig_obs), axis=1)              # (B,)

    resid = ig_obs - cgm[None, :]
    logL = (-0.5 * np.sum((resid / sigma) ** 2, axis=1)
            - n * np.log(sigma))
    return np.where(finite_mask, logL, -np.inf)


# ---------------------------------------------------------------------------
# Joint log-probability (vectorised callable: log-prior + log-likelihood)
# ---------------------------------------------------------------------------

def make_log_prob(
    cgm: np.ndarray,
    insulin: np.ndarray,
    cho: np.ndarray,
    Ib: float,
    sample_time: float,
    dt: float = 1.0,
    sigma: float = DEFAULT_SIGMA,
):
    """Return a vectorised ``log_prob(phi_batch)`` callable, shape (B, 8) -> (B,)."""
    def log_prob(phi_batch: np.ndarray) -> np.ndarray:
        """Unnormalised log posterior over a batch of phi vectors."""
        phi_batch = np.atleast_2d(phi_batch)
        lp = log_prior(phi_batch)
        out = np.full(lp.shape, -np.inf)
        in_prior = np.isfinite(lp)
        if np.any(in_prior):
            ll = log_likelihood(
                phi_batch[in_prior],
                cgm, insulin, cho, Ib, sample_time, dt, sigma,
            )
            out[in_prior] = lp[in_prior] + ll
        return out

    return log_prob


# ---------------------------------------------------------------------------
# Walker initialisation
# ---------------------------------------------------------------------------

def _init_chain(Gb_init: float, rng: np.random.Generator) -> np.ndarray:
    """Initial chain state (phi, shape (8,)) at the prior centre."""
    pop_theta = theta_to_array(_priors.prior_center()).copy()
    pop_theta[7] = float(np.clip(Gb_init, PRIOR_LO[7] * 1.01, PRIOR_HI[7] * 0.99))
    pop_theta = np.clip(pop_theta, PRIOR_LO * 1.01, PRIOR_HI * 0.99)
    phi = _theta_to_phi(pop_theta) + 0.01 * rng.standard_normal(8)
    theta = np.clip(_phi_to_theta(phi), PRIOR_LO * 1.01, PRIOR_HI * 0.99)
    return _theta_to_phi(theta)


# ---------------------------------------------------------------------------
# Adaptive Single-Component Metropolis-Hastings (ReplayBG sampler)
# ---------------------------------------------------------------------------

def _single_chain_ascmh(
    log_prob_fn,
    phi0: np.ndarray,
    nburn: int,
    nsample: int,
    rng: np.random.Generator,
    target_accept: float = TARGET_ACCEPT,
    adapt_batch: int = ADAPT_BATCH,
    progress: bool = False,
) -> np.ndarray:
    """Adaptive Metropolis-within-Gibbs chain; returns (nsample, 8) phi samples."""
    phi = np.asarray(phi0, dtype=float).copy()
    cur_lp = float(log_prob_fn(phi[None, :])[0])
    # If we start outside the support (should not happen from _init_chain), nudge
    # until finite.
    tries = 0
    while not np.isfinite(cur_lp) and tries < 100:
        phi = phi + 0.05 * rng.standard_normal(8)
        cur_lp = float(log_prob_fn(phi[None, :])[0])
        tries += 1

    log_step = np.log(_INIT_STEP).copy()
    acc = np.zeros(8)
    batch_k = 0
    total = nburn + nsample
    out = np.empty((nsample, 8))

    for it in range(total):
        for j in range(8):
            phi_prop = phi.copy()
            phi_prop[j] += np.exp(log_step[j]) * rng.standard_normal()
            prop_lp = float(log_prob_fn(phi_prop[None, :])[0])
            if np.log(rng.random()) < prop_lp - cur_lp:
                phi = phi_prop
                cur_lp = prop_lp
                acc[j] += 1.0

        # Diminishing adaptation during burn-in only.
        if it < nburn and (it + 1) % adapt_batch == 0:
            batch_k += 1
            delta = min(0.01, 1.0 / np.sqrt(batch_k))
            rate = acc / adapt_batch
            log_step += np.where(rate > target_accept, delta, -delta)
            acc[:] = 0.0

        if it >= nburn:
            out[it - nburn] = phi

        if progress and (it + 1) % max(1, total // 20) == 0:
            print(f"    [mcmc] sweep {it + 1}/{total}  logpost={cur_lp:.1f}")

    return out


# ---------------------------------------------------------------------------
# MCMC driver
# ---------------------------------------------------------------------------

def run_mcmc(
    cgm: np.ndarray,
    insulin: np.ndarray,
    cho: np.ndarray,
    Ib: float,
    sample_time: float,
    dt: float = 1.0,
    sigma: float = DEFAULT_SIGMA,
    nburn: int = DEFAULT_BURN,
    nsample: int = DEFAULT_NSAMPLE,
    n_posterior: int = DEFAULT_N_POSTERIOR,
    seed: int = 42,
    progress: bool = _PROGRESS_DEFAULT,
    nwalkers: Optional[int] = None,   # deprecated, ignored (single-chain sampler)
) -> np.ndarray:
    """Run the ReplayBG MCMC pipeline; return ``n_posterior`` posterior θ samples."""
    rng = np.random.default_rng(seed)

    Gb_init = float(np.median(cgm[:max(1, len(cgm) // 4)]))
    phi0 = _init_chain(Gb_init, rng)

    log_prob_fn = make_log_prob(cgm, insulin, cho, Ib, sample_time, dt, sigma)

    # Adaptive-scaling component-wise Metropolis-Hastings in phi-space.
    phi_chain = _single_chain_ascmh(
        log_prob_fn, phi0, nburn, nsample, rng, progress=progress,
    )
    theta_chain = _phi_to_theta(phi_chain)                       # (nsample, 8)

    # Guard: if the chain never moved (degenerate), fall back to the raw samples.
    if np.allclose(theta_chain.std(axis=0), 0.0):
        warnings.warn(
            "MCMC chain did not mix (zero variance); returning raw samples "
            "without copula smoothing.", RuntimeWarning, stacklevel=2)
        idx = rng.integers(0, len(theta_chain), size=n_posterior)
        return theta_chain[idx]

    # Resample from a t-copula fitted to the chain, keeping admissible draws.
    copula = fit_t_copula(theta_chain)
    kept: list[np.ndarray] = []
    attempts = 0
    while sum(len(k) for k in kept) < n_posterior and attempts < 50:
        draws = copula.sample(max(2 * n_posterior, 256), rng)
        ok = _valid_theta_mask(draws)
        kept.append(draws[ok])
        attempts += 1
    valid = (np.vstack(kept) if kept else np.empty((0, 8)))
    if len(valid) >= n_posterior:
        theta_post = valid[:n_posterior]
    else:
        # Copula rarely produced in-support draws — top up from the raw chain.
        warnings.warn(
            f"t-copula yielded only {len(valid)}/{n_posterior} in-support "
            "realizations; topping up from the raw chain.",
            RuntimeWarning, stacklevel=2)
        need = n_posterior - len(valid)
        idx = rng.integers(0, len(theta_chain), size=need)
        theta_post = np.vstack([valid, theta_chain[idx]]) if len(valid) else theta_chain[idx]

    return theta_post


def _valid_theta_mask(theta: np.ndarray) -> np.ndarray:
    """Boolean mask of θ rows inside the support box AND satisfying the ordering."""
    theta = np.atleast_2d(theta)
    in_box = np.all((theta >= PRIOR_LO) & (theta <= PRIOR_HI), axis=1)
    order = (theta[:, 0] < theta[:, 1]) & (theta[:, 3] < theta[:, 2])   # ka2<kd, kabs<kempt
    return in_box & order


# ---------------------------------------------------------------------------
# MCMCTwin — implements the common Twin interface
# ---------------------------------------------------------------------------

class MCMCTwin(Twin):
    """ReplayBG digital twin identified by Bayesian MCMC."""

    def __init__(
        self,
        theta_post: np.ndarray,
        Ib: float,
        sample_time: float,
        dt: float = 1.0,
        sensor_name: str = "Dexcom",
        sigma: float = DEFAULT_SIGMA,
    ) -> None:
        self.theta_post: np.ndarray = np.asarray(theta_post, dtype=float)
        self.theta_median: np.ndarray = np.median(self.theta_post, axis=0)
        self.Ib = float(Ib)
        self.sample_time = float(sample_time)
        self.dt = float(dt)
        self.sensor_name = sensor_name
        self.sigma = float(sigma)

    # ------------------------------------------------------------------
    # Twin ABC implementation
    # ------------------------------------------------------------------

    def predict_ig(
        self,
        insulin: np.ndarray,
        cho: np.ndarray,
        n_samples: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate IG ensemble over the posterior, return (median, IQR)."""
        ig_ens = self._ig_ensemble(insulin, cho, n_samples)
        ig_median = np.median(ig_ens, axis=0)
        ig_iqr = np.percentile(ig_ens, [25, 75], axis=0)
        return ig_median, ig_iqr

    def _ig_ensemble(
        self,
        insulin: np.ndarray,
        cho: np.ndarray,
        n_samples: Optional[int] = None,
    ) -> np.ndarray:
        """Return (R, T) IG ensemble from batched posterior simulate."""
        theta_b = self.theta_post
        if n_samples is not None and n_samples < len(theta_b):
            idx = np.random.default_rng(0).choice(
                len(theta_b), size=n_samples, replace=False
            )
            theta_b = theta_b[idx]
        _, ig_ens = simulate(theta_b, insulin, cho, Ib=self.Ib, dt=self.dt)
        if n_samples is not None and n_samples > len(ig_ens):
            idx = np.random.default_rng(0).choice(
                len(ig_ens), size=n_samples, replace=True)
            ig_ens = ig_ens[idx]
        return ig_ens   # (R, T)

    def summary(self) -> dict:
        """Per-parameter posterior medians and 95% credible intervals."""
        result = {}
        for i, name in enumerate(THETA_NAMES):
            col = self.theta_post[:, i]
            lo, med, hi = np.percentile(col, [2.5, 50, 97.5])
            result[name] = {
                "median": float(med),
                "ci95": (float(lo), float(hi)),
            }
        return result

    # ------------------------------------------------------------------
    # Extra MCMC-specific properties
    # ------------------------------------------------------------------

    def n_posterior(self) -> int:
        """Number of posterior samples stored."""
        return self.theta_post.shape[0]


# ---------------------------------------------------------------------------
# High-level glue
# ---------------------------------------------------------------------------

def identify_twin_from_run(
    run: RunResult,
    dt: float = 1.0,
    sigma: float = DEFAULT_SIGMA,
    sensor_name: str = "Dexcom",
    nburn: int = DEFAULT_BURN,
    nsample: int = DEFAULT_NSAMPLE,
    n_posterior: int = DEFAULT_N_POSTERIOR,
    seed: int = 42,
    progress: bool = _PROGRESS_DEFAULT,
    nwalkers: Optional[int] = None,   # deprecated no-op (single-chain sampler)
) -> MCMCTwin:
    """Identify an MCMCTwin from a simglucose ``RunResult``."""
    cgm = run.cgm()
    _, insulin, cho = run.replaybg_inputs(dt=dt)
    Ib = float(insulin[0])

    theta_post = run_mcmc(
        cgm=cgm,
        insulin=insulin,
        cho=cho,
        Ib=Ib,
        sample_time=run.sample_time,
        dt=dt,
        sigma=sigma,
        nburn=nburn,
        nsample=nsample,
        n_posterior=n_posterior,
        seed=seed,
        progress=progress,
    )

    return MCMCTwin(
        theta_post=theta_post,
        Ib=Ib,
        sample_time=run.sample_time,
        dt=dt,
        sensor_name=sensor_name,
        sigma=sigma,
    )