"""SBI (Simulation-Based Inference) twin identification (Hoang et al., 2025)."""
from __future__ import annotations

import os
import warnings
from typing import Optional, Tuple

import numpy as np
import torch

#: Intra-op torch threads used while training the MAF.
TORCH_THREADS: int = 1

_VERBOSE_DEFAULT = os.environ.get("T1D_VERBOSE") == "1"

from sbi.inference import SNPE, simulate_for_sbi

from .replaybg_model import (
    THETA_NAMES,
    simulate, sample_indices, steady_state, theta_to_array,
)
from . import replaybg_priors as _priors
from .simglucose_adapter import RunResult
from .twin import Twin

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

# Prior bounds — same as MCMC A2.
PRIOR_LO = _priors.SUPPORT_LO.copy()
PRIOR_HI = _priors.SUPPORT_HI.copy()

# Default training-data set size (paper: 5 000)
N_TRAIN_DEFAULT: int = 5_000

# CGM physiological rejection bounds [mg/dL]
CGM_LO: float = 40.0
CGM_HI: float = 400.0

# Noise std added to simulator output during training [mg/dL] Matches Dexcom empirical
# std used in MCMC A2.
DEFAULT_SIGMA: float = 10.0

# Default NPE posterior samples drawn per identification
DEFAULT_N_POSTERIOR: int = 1_000

# Fixed log-space transforms for rate params (indices 0-6) same as MCMC
_LOG_IDX = list(range(_priors.N_LOG_RATES))   # leading rate params are log-transformed


# ---------------------------------------------------------------------------
# Log-space <-> natural transforms (same convention as MCMC)
# ---------------------------------------------------------------------------
# Canonical definitions live in ``replaybg_priors`` (the single phi-convention
# authority).

_theta_to_phi = _priors.theta_to_phi
_phi_to_theta = _priors.phi_to_theta
theta_to_phi_torch = _priors.theta_to_phi_torch
phi_to_theta_torch = _priors.phi_to_theta_torch


# ---------------------------------------------------------------------------
# A3 — Part 1: Prior
# ---------------------------------------------------------------------------

def make_prior(log_space: bool = True, seed: int | None = None):
    """Return the informative published (Cappon et al.) prior over phi-space."""
    if not log_space:
        raise NotImplementedError(
            "The published prior is defined in phi (log) space; "
            "call make_prior(log_space=True).")
    prior = _priors.make_sbi_prior(seed=seed)
    # Best-effort adapt to the installed sbi version; the raw object already exposes
    # .sample/.log_prob if process_prior is unavailable.
    try:
        from sbi.utils.user_input_checks import process_prior
        prior, _, _ = process_prior(prior)
    except Exception:
        pass
    return prior


# ---------------------------------------------------------------------------
# A3 — Part 2: Simulator
# ---------------------------------------------------------------------------

def make_simulator(
    insulin: np.ndarray,
    cho: np.ndarray,
    Ib: float,
    sample_time: float,
    dt: float = 1.0,
    sigma: float = DEFAULT_SIGMA,
    log_space: bool = True,
    seed: Optional[int] = None,
):
    """Build a ``sbi``-compatible simulator wrapping the ReplayBG ODE."""
    T = len(insulin)
    n_obs = T // int(round(sample_time / dt))
    obs_idx = sample_indices(n_obs, sample_time, dt)

    rng_state = np.random.default_rng(seed)

    def simulator(phi_t: torch.Tensor) -> torch.Tensor:
        """Map a batch of phi to simulated observation vectors for sbi."""
        phi = phi_t.detach().cpu().numpy().astype(float)
        theta = _phi_to_theta(phi) if log_space else phi

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                _, ig = simulate(theta, insulin, cho, Ib=Ib, dt=dt)
            except Exception:
                return torch.full((n_obs,), float("nan"), dtype=torch.float32)

        if not np.all(np.isfinite(ig)):
            return torch.full((n_obs,), float("nan"), dtype=torch.float32)

        ig_obs = ig[obs_idx]
        if sigma > 0.0:
            ig_obs = ig_obs + rng_state.normal(0.0, sigma, size=n_obs)
        return torch.tensor(ig_obs, dtype=torch.float32)

    return simulator, n_obs


# ---------------------------------------------------------------------------
# A3 — Part 3: Training-data generation with rejection sampling
# ---------------------------------------------------------------------------

def generate_training_data(
    insulin: np.ndarray,
    cho: np.ndarray,
    Ib: float,
    sample_time: float,
    dt: float = 1.0,
    sigma: float = DEFAULT_SIGMA,
    n_train: int = N_TRAIN_DEFAULT,
    cgm_lo: float = CGM_LO,
    cgm_hi: float = CGM_HI,
    log_space: bool = True,
    seed: int = 42,
    batch_size: int = 512,
    max_attempts: int = 20,
    verbose: bool = _VERBOSE_DEFAULT,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate (θ, y) training pairs with rejection sampling."""
    rng_phi = np.random.default_rng(seed)
    rng_noise = np.random.default_rng(seed + 1)

    # seed the prior itself — prior.sample() is what draws the training thetas
    # (rng_phi below is vestigial; see finding C4).
    prior = make_prior(log_space=log_space, seed=seed)
    simulator, n_obs = make_simulator(
        insulin, cho, Ib, sample_time, dt, sigma=0.0,  # we add noise below
        log_space=log_space, seed=seed + 2,
    )

    T = len(insulin)
    obs_idx = sample_indices(n_obs, sample_time, dt)

    accepted_phi: list[np.ndarray] = []
    accepted_obs: list[np.ndarray] = []
    n_accepted = 0
    n_attempts = 0

    if verbose:
        print(f"  [SBI A3] Generating {n_train} training pairs "
              f"(rejection bounds [{cgm_lo}, {cgm_hi}] mg/dL) ...")

    while n_accepted < n_train and n_attempts < max_attempts:
        n_attempts += 1
        # Draw batch of phi from prior
        phi_batch_t = prior.sample((batch_size,))          # (B, 8)
        phi_batch = phi_batch_t.numpy()                    # (B, 8)

        # Convert to theta for batched simulate
        theta_batch = _phi_to_theta(phi_batch) if log_space else phi_batch

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                _, ig_batch = simulate(theta_batch, insulin, cho,
                                       Ib=Ib, dt=dt)     # (B, T)
            except Exception as exc:
                if verbose:
                    print(f"  [warn] simulate batch failed: {exc}")
                continue

        ig_obs_batch = ig_batch[:, obs_idx]                # (B, n_obs)

        # Add sensor noise
        if sigma > 0.0:
            noise = rng_noise.normal(0.0, sigma,
                                     size=ig_obs_batch.shape)
            ig_obs_batch = ig_obs_batch + noise

        # Rejection filter: physiological range + finite
        finite_mask = np.all(np.isfinite(ig_obs_batch), axis=1)
        range_mask = (
            np.all(ig_obs_batch >= cgm_lo, axis=1) &
            np.all(ig_obs_batch <= cgm_hi, axis=1)
        )
        keep = finite_mask & range_mask                    # (B,)

        n_kept = int(keep.sum())
        if n_kept == 0:
            continue

        accepted_phi.append(phi_batch[keep])
        accepted_obs.append(ig_obs_batch[keep])
        n_accepted += n_kept

        if verbose:
            accept_rate = n_kept / batch_size * 100.0
            print(f"    round {n_attempts:2d}: kept {n_kept}/{batch_size} "
                  f"({accept_rate:.1f}%),  total {min(n_accepted, n_train)}/{n_train}")

        if n_accepted >= n_train:
            break

    if n_accepted < n_train:
        warnings.warn(
            f"Only {n_accepted}/{n_train} training samples collected after "
            f"{n_attempts} rounds (each of size {batch_size}). "
            "Consider increasing max_attempts or relaxing rejection bounds.",
            RuntimeWarning, stacklevel=2,
        )

    phi_all = np.vstack(accepted_phi)[:n_train]
    obs_all = np.vstack(accepted_obs)[:n_train]

    theta_tensor = torch.tensor(phi_all, dtype=torch.float32)
    obs_tensor = torch.tensor(obs_all, dtype=torch.float32)

    if verbose:
        print(f"  [SBI A3] Done: {len(theta_tensor)} pairs collected.")

    return theta_tensor, obs_tensor


# ---------------------------------------------------------------------------
# A3 — Part 4: Joint initial-condition inference (Hoang et al.
# ---------------------------------------------------------------------------
# The paper's headline contribution: instead of assuming steady-state initial
# conditions, treat the 9 model states x0 as parameters and infer them.

N_STATES = 9   # [Isc1, Isc2, Ip, Qsto1, Qsto2, Qgut, G, X, IG]


def sample_x0_given_theta(
    theta_nat: np.ndarray,
    insulin: np.ndarray,
    cho: np.ndarray,
    Ib: float,
    sample_time: float,
    dt: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample x0 ~ p(x0|theta) by the ReplayBG burn-in procedure (Hoang §3.1)."""
    if rng is None:
        rng = np.random.default_rng()
    theta_nat = np.atleast_2d(np.asarray(theta_nat, dtype=float))
    B = theta_nat.shape[0]
    T = len(insulin)

    # --- phase 1: burn-in on the record itself, stopping at a random index -----
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _, _, burn_states = simulate(theta_nat, insulin, cho, Ib=Ib, dt=dt,
                                     return_states=True)          # (B, T, 9)
    burn_states = np.asarray(burn_states)
    if burn_states.ndim == 2:                                     # single-theta squeeze
        burn_states = burn_states[None]

    # offset 0 => steady state (the state before interval 0); offset k>=1 => the state
    # at the end of interval k-1.
    offsets = rng.integers(0, T, size=B)
    x0 = np.atleast_2d(steady_state(theta_nat, Ib)).astype(float).copy()   # (B,9)
    nz = offsets >= 1
    if np.any(nz):
        x0[nz] = burn_states[np.nonzero(nz)[0], offsets[nz] - 1, :]

    # --- phase 2: re-simulate the SAME record from x0, inputs un-rotated -------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _, _, obs_states = simulate(theta_nat, insulin, cho, Ib=Ib, dt=dt,
                                    x0=x0, return_states=True)    # (B, T, 9)
    obs_states = np.asarray(obs_states)
    if obs_states.ndim == 2:
        obs_states = obs_states[None]

    n_obs = T // int(round(sample_time / dt))
    obs_idx = sample_indices(n_obs, sample_time, dt)              # indices in [0, T)
    y_ig = obs_states[:, obs_idx, 8]                              # IG channel
    return x0, y_ig


def generate_training_data_ic(
    insulin: np.ndarray,
    cho: np.ndarray,
    Ib: float,
    sample_time: float,
    dt: float = 1.0,
    sigma: float = DEFAULT_SIGMA,
    n_train: int = N_TRAIN_DEFAULT,
    cgm_lo: float = CGM_LO,
    cgm_hi: float = CGM_HI,
    seed: int = 42,
    batch_size: int = 512,
    max_attempts: int = 400,
    verbose: bool = _VERBOSE_DEFAULT,
) -> Tuple["torch.Tensor", "torch.Tensor", np.ndarray, np.ndarray]:
    """Training pairs for joint IC inference: theta-hat = [phi(8), x0(9)] and y."""
    rng_phi = np.random.default_rng(seed)
    rng_x0 = np.random.default_rng(seed + 3)
    rng_noise = np.random.default_rng(seed + 1)

    # seed the prior itself — prior.sample() is what draws the training thetas
    # (rng_phi below is vestigial; see finding C4).
    prior = make_prior(log_space=True, seed=seed)
    T = len(insulin)
    n_obs = T // int(round(sample_time / dt))

    kept_theta_hat: list[np.ndarray] = []
    kept_obs: list[np.ndarray] = []
    n_acc = 0
    attempts = 0
    if verbose:
        print(f"  [SBI A3-IC] Generating {n_train} (theta-hat, y) pairs with "
              f"joint IC sampling (rejection [{cgm_lo},{cgm_hi}] mg/dL) ...")

    while n_acc < n_train and attempts < max_attempts:
        attempts += 1
        phi_batch = prior.sample((batch_size,)).numpy()               # (B, 8)
        theta_batch = _phi_to_theta(phi_batch)                        # natural
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                x0, y_ig = sample_x0_given_theta(
                    theta_batch, insulin, cho, Ib, sample_time, dt, rng_x0)
            except Exception as exc:                                   # pragma: no cover
                if verbose:
                    print(f"  [warn] x0 sampling failed: {exc}")
                continue
        if sigma > 0.0:
            y_ig = y_ig + rng_noise.normal(0.0, sigma, size=y_ig.shape)

        finite = np.all(np.isfinite(y_ig), axis=1) & np.all(np.isfinite(x0), axis=1)
        in_range = np.all(y_ig >= cgm_lo, axis=1) & np.all(y_ig <= cgm_hi, axis=1)
        keep = finite & in_range
        if not np.any(keep):
            continue
        theta_hat = np.concatenate([phi_batch[keep], x0[keep]], axis=1)   # (k,17)
        kept_theta_hat.append(theta_hat)
        kept_obs.append(y_ig[keep])
        n_acc += int(keep.sum())

    if n_acc == 0:
        raise RuntimeError("IC training-data generation accepted 0 samples.")
    theta_hat_all = np.vstack(kept_theta_hat)[:n_train]
    obs_all = np.vstack(kept_obs)[:n_train]
    x0_lo = theta_hat_all[:, 8:].min(axis=0)
    x0_hi = theta_hat_all[:, 8:].max(axis=0)

    if n_acc < n_train:
        warnings.warn(
            f"Only {n_acc}/{n_train} IC training samples collected after "
            f"{attempts} rounds.", RuntimeWarning, stacklevel=2)
    if verbose:
        print(f"  [SBI A3-IC] Done: {len(theta_hat_all)} pairs.")

    theta_tensor = torch.tensor(theta_hat_all, dtype=torch.float32)
    obs_tensor = torch.tensor(obs_all, dtype=torch.float32)
    return theta_tensor, obs_tensor, x0_lo, x0_hi


def make_prior_ic(x0_lo: np.ndarray, x0_hi: np.ndarray):
    """17-dim support-box prior for joint IC inference: [phi support] x [x0 box]."""
    from sbi.utils import BoxUniform
    phi_lo = _theta_to_phi(PRIOR_LO.copy())
    phi_hi = _theta_to_phi(PRIOR_HI.copy())
    pad = 0.10 * (np.asarray(x0_hi) - np.asarray(x0_lo)) + 1e-6
    low = np.concatenate([phi_lo, np.asarray(x0_lo) - pad]).astype(np.float32)
    high = np.concatenate([phi_hi, np.asarray(x0_hi) + pad]).astype(np.float32)
    return BoxUniform(low=torch.tensor(low), high=torch.tensor(high))


# ---------------------------------------------------------------------------
# A4 — Part 1: Train NPE (MAF)
# ---------------------------------------------------------------------------

def train_npe(
    theta_tensor: torch.Tensor,
    obs_tensor: torch.Tensor,
    prior,
    hidden_features: int = 50,
    num_transforms: int = 5,
    batch_size: int = 200,
    learning_rate: float = 5e-4,
    clip_max_norm: float = 5.0,
    validation_fraction: float = 0.10,
    stop_after_epochs: int = 20,
    max_num_epochs: int = 200,
    verbose: bool = _VERBOSE_DEFAULT,
    seed: int | None = None,
    torch_threads: int | None = TORCH_THREADS,
) -> object:
    """Train a Masked Autoregressive Flow (MAF) posterior estimator."""
    if verbose:
        print(f"  [SBI A4] Training NPE (MAF) on {len(theta_tensor)} samples ...")

    # The sbi library writes TensorBoard logs via a SummaryWriter that, by default,
    # lands in a `sbi_logs`/`sbi-logs` folder in the current working.
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        root = os.environ.get("T1D_OUTPUT_ROOT", os.path.join(os.getcwd(), "_out"))
        if os.environ.get("SLURM_ARRAY_JOB_ID"):
            tag = f"{os.environ['SLURM_ARRAY_JOB_ID']}_{os.environ.get('SLURM_ARRAY_TASK_ID', '0')}"
        elif os.environ.get("SLURM_JOB_ID"):
            tag = os.environ["SLURM_JOB_ID"]
        else:
            from datetime import datetime
            tag = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        log_dir = os.path.join(root, "sbi_logs", tag)
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
    except Exception:
        writer = None  # fall back to sbi's default location if TB is unavailable

    try:
        from sbi.neural_nets import posterior_nn
    except Exception:                       # older sbi layout
        from sbi.utils import posterior_nn
    _density_estimator = posterior_nn(
        model="maf",
        hidden_features=hidden_features,
        num_transforms=num_transforms,
    )
    try:
        inference = SNPE(prior=prior, density_estimator=_density_estimator,
                         summary_writer=writer)
    except TypeError:
        # very different sbi version without the summary_writer kwarg
        inference = SNPE(prior=prior, density_estimator=_density_estimator)
    inference = inference.append_simulations(theta_tensor, obs_tensor)

    _prev_threads = torch.get_num_threads()
    if torch_threads is not None:
        torch.set_num_threads(int(torch_threads))
    if seed is not None:
        torch.manual_seed(int(seed))
    try:
        density_estimator = inference.train(
            training_batch_size=batch_size,
            learning_rate=learning_rate,
            clip_max_norm=clip_max_norm,
            validation_fraction=validation_fraction,
            stop_after_epochs=stop_after_epochs,
            max_num_epochs=max_num_epochs,
            show_train_summary=verbose,
        )
        posterior = inference.build_posterior(density_estimator)
    finally:
        torch.set_num_threads(_prev_threads)
    if writer is not None:
        writer.close()
    if verbose:
        print("  [SBI A4] NPE training complete.")
    return posterior


# ---------------------------------------------------------------------------
# A4 — Part 2: SBITwin wrapper
# ---------------------------------------------------------------------------

class SBITwin(Twin):
    """ReplayBG digital twin identified by Simulation-Based Inference (NPE/MAF)."""

    def __init__(
        self,
        posterior,
        theta_post: np.ndarray,
        Ib: float,
        sample_time: float,
        dt: float = 1.0,
        log_space: bool = True,
        sensor_name: str = "Dexcom",
        sigma: float = DEFAULT_SIGMA,
        x0_post: Optional[np.ndarray] = None,
    ) -> None:
        self._posterior = posterior
        # Store phi_post (log-space) internally; expose natural theta for summaries
        phi_post = np.asarray(theta_post, dtype=float)
        self._phi_post = phi_post
        self.theta_post = _phi_to_theta(phi_post) if log_space else phi_post
        self.theta_median = np.median(self.theta_post, axis=0)
        self.Ib = float(Ib)
        self.sample_time = float(sample_time)
        self.dt = float(dt)
        self.log_space = log_space
        self.sensor_name = sensor_name
        self.sigma = float(sigma)
        # Joint IC inference (Hoang §3.1): per-posterior-sample inferred initial
        # states.
        self.x0_post: Optional[np.ndarray] = (
            np.asarray(x0_post, dtype=float) if x0_post is not None else None)

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
        x0_b = self.x0_post
        if n_samples is not None and n_samples < len(theta_b):
            idx = np.random.default_rng(0).choice(
                len(theta_b), size=n_samples, replace=False,
            )
            theta_b = theta_b[idx]
            if x0_b is not None:
                x0_b = x0_b[idx]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            # With joint IC inference, replay from the inferred per-sample x0 (Hoang
            # §3.1); otherwise fall back to steady-state ICs.
            _, ig_ens = simulate(theta_b, insulin, cho, Ib=self.Ib, dt=self.dt,
                                  x0=x0_b)
        # Replace non-finite rows with column medians
        bad = ~np.all(np.isfinite(ig_ens), axis=1)
        if np.any(bad):
            col_med = np.nanmedian(ig_ens[~bad], axis=0) if np.any(~bad) else np.full(ig_ens.shape[1], 120.0)
            ig_ens[bad] = col_med
        # Resample with replacement when more rows are requested than exist.
        if n_samples is not None and n_samples > len(ig_ens):
            idx = np.random.default_rng(0).choice(
                len(ig_ens), size=n_samples, replace=True)
            ig_ens = ig_ens[idx]
        return ig_ens  # (R, T)

    def summary(self) -> dict:
        """Per-parameter posterior medians and 95% credible intervals (natural θ)."""
        result = {}
        for i, name in enumerate(THETA_NAMES):
            col = self.theta_post[:, i]
            lo, med, hi = np.percentile(col, [2.5, 50, 97.5])
            result[name] = {
                "median": float(med),
                "ci95": (float(lo), float(hi)),
            }
        return result

    def n_posterior(self) -> int:
        """Number of posterior samples stored."""
        return self.theta_post.shape[0]


# ---------------------------------------------------------------------------
# A4 — Part 3: identify_twin_from_run (high-level glue)
# ---------------------------------------------------------------------------

def identify_twin_from_run(
    run: RunResult,
    dt: float = 1.0,
    sigma: float = DEFAULT_SIGMA,
    sensor_name: str = "Dexcom",
    n_train: int = N_TRAIN_DEFAULT,
    n_posterior: int = DEFAULT_N_POSTERIOR,
    log_space: bool = True,
    hidden_features: int = 50,
    num_transforms: int = 5,
    batch_size_train: int = 200,
    learning_rate: float = 5e-4,
    clip_max_norm: float = 5.0,
    validation_fraction: float = 0.10,
    stop_after_epochs: int = 20,
    max_num_epochs: int = 200,
    seed: int = 42,
    infer_ic: bool = False,
    verbose: bool = _VERBOSE_DEFAULT,
) -> "SBITwin":
    """Identify an SBITwin from a simglucose ``RunResult``."""
    cgm = run.cgm()
    _, insulin, cho = run.replaybg_inputs(dt=dt)
    Ib = float(insulin[0])

    if infer_ic:
        return _identify_twin_ic(
            cgm=cgm, insulin=insulin, cho=cho, Ib=Ib,
            sample_time=run.sample_time, dt=dt, sigma=sigma,
            sensor_name=sensor_name, n_train=n_train, n_posterior=n_posterior,
            hidden_features=hidden_features, num_transforms=num_transforms,
            batch_size_train=batch_size_train, learning_rate=learning_rate,
            clip_max_norm=clip_max_norm, validation_fraction=validation_fraction,
            stop_after_epochs=stop_after_epochs, max_num_epochs=max_num_epochs,
            seed=seed, verbose=verbose,
        )

    # --- build prior ---
    prior = make_prior(log_space=log_space)

    # --- generate training data ---
    theta_tensor, obs_tensor = generate_training_data(
        insulin=insulin,
        cho=cho,
        Ib=Ib,
        sample_time=run.sample_time,
        dt=dt,
        sigma=sigma,
        n_train=n_train,
        log_space=log_space,
        seed=seed,
        verbose=verbose,
    )

    # --- train NPE ---
    posterior = train_npe(
        theta_tensor=theta_tensor,
        obs_tensor=obs_tensor,
        prior=prior,
        hidden_features=hidden_features,
        num_transforms=num_transforms,
        batch_size=batch_size_train,
        learning_rate=learning_rate,
        clip_max_norm=clip_max_norm,
        validation_fraction=validation_fraction,
        stop_after_epochs=stop_after_epochs,
        max_num_epochs=max_num_epochs,
        verbose=verbose,
        seed=seed,
    )

    # --- condition posterior on observed CGM ---
    # The observation vector is the full CGM as a flat Tensor
    y_obs = torch.tensor(cgm, dtype=torch.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        posterior_samples = posterior.sample(
            (n_posterior,), x=y_obs, show_progress_bars=verbose,
            reject_outside_prior=False,
        )  # (n_posterior, 8) in phi-space

    phi_post = posterior_samples.detach().cpu().numpy()

    return SBITwin(
        posterior=posterior,
        theta_post=phi_post,
        Ib=Ib,
        sample_time=run.sample_time,
        dt=dt,
        log_space=log_space,
        sensor_name=sensor_name,
        sigma=sigma,
    )


def _identify_twin_ic(
    cgm: np.ndarray,
    insulin: np.ndarray,
    cho: np.ndarray,
    Ib: float,
    sample_time: float,
    dt: float,
    sigma: float,
    sensor_name: str,
    n_train: int,
    n_posterior: int,
    hidden_features: int,
    num_transforms: int,
    batch_size_train: int,
    learning_rate: float,
    clip_max_norm: float,
    validation_fraction: float,
    stop_after_epochs: int,
    max_num_epochs: int,
    seed: int,
    verbose: bool,
) -> "SBITwin":
    """Joint IC-inference pipeline (Hoang §3.1): infer θ̂ = [phi(8), x0(9)] ∈ R¹⁷."""
    theta_hat_t, obs_t, x0_lo, x0_hi = generate_training_data_ic(
        insulin=insulin, cho=cho, Ib=Ib, sample_time=sample_time, dt=dt,
        sigma=sigma, n_train=n_train, seed=seed, verbose=verbose,
    )
    prior = make_prior_ic(x0_lo, x0_hi)

    posterior = train_npe(
        theta_tensor=theta_hat_t,
        obs_tensor=obs_t,
        prior=prior,
        hidden_features=hidden_features,
        num_transforms=num_transforms,
        batch_size=batch_size_train,
        learning_rate=learning_rate,
        clip_max_norm=clip_max_norm,
        validation_fraction=validation_fraction,
        stop_after_epochs=stop_after_epochs,
        max_num_epochs=max_num_epochs,
        verbose=verbose,
        seed=seed,
    )

    y_obs = torch.tensor(cgm, dtype=torch.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        theta_hat_post = posterior.sample(
            (n_posterior,), x=y_obs, show_progress_bars=verbose,
            reject_outside_prior=False,
        ).detach().cpu().numpy()                       # (n_posterior, 17)

    phi_post = theta_hat_post[:, :8]                    # log-space params
    x0_post = theta_hat_post[:, 8:]                     # inferred initial states

    # The paper's prior is support- truncated: its reference implementation's
    # CustomPrior.log_prob assigns -inf to any draw with a negative initial state.
    _th = np.atleast_2d(_phi_to_theta(phi_post))
    _valid = (
        np.all(x0_post >= 0.0, axis=1)                                   # no negative states
        & np.all((_th >= PRIOR_LO) & (_th <= PRIOR_HI), axis=1)          # support box
        & (_th[:, 0] < _th[:, 1]) & (_th[:, 3] < _th[:, 2])              # ka2<kd, kabs<kempt
    )
    _n_ok = int(np.count_nonzero(_valid))
    if 0 < _n_ok < len(_valid):
        _idx = np.nonzero(_valid)[0]
        _take = np.random.default_rng(seed).choice(_idx, size=len(_valid), replace=True)
        phi_post, x0_post = phi_post[_take], x0_post[_take]
        if verbose:
            print(f"  [SBI] rejected {len(_valid) - _n_ok}/{len(_valid)} draws "
                  f"outside the paper's support (negative x0 or ordering); resampled.")
    elif _n_ok == 0:
        print("  [SBI][warn] EVERY posterior draw violates the paper's support "
              "(negative x0 or ordering) — keeping raw draws; this twin is suspect.")

    return SBITwin(
        posterior=posterior,
        theta_post=phi_post,
        Ib=Ib,
        sample_time=sample_time,
        dt=dt,
        log_space=True,
        sensor_name=sensor_name,
        sigma=sigma,
        x0_post=x0_post,
    )