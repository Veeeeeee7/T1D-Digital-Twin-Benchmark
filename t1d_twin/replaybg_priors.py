"""Published ReplayBG priors (Cappon et al., 2023) — single source of truth."""
from __future__ import annotations

import numpy as np
from scipy.special import gammaln, erf

from .replaybg_model import THETA_NAMES, VG

# ---------------------------------------------------------------------------
# Parameter indices (kept in sync with THETA_NAMES)
# ---------------------------------------------------------------------------
I_KA2, I_KD, I_KEMPT, I_KABS, I_SG, I_SI, I_P2, I_GB = range(8)

# Indices whose prior is a plain Normal in phi (= LogNormal in theta).
_LOGNORMAL_RATE_IDX = (I_KA2, I_KD, I_KEMPT, I_KABS, I_SG)


# ---------------------------------------------------------------------------
# Canonical coordinate transform: theta (natural) <-> phi (sampler space)
# ---------------------------------------------------------------------------
# ``phi`` is the space every identification method infers in: the leading
# ``N_LOG_RATES`` rate constants are carried in LOG space.
N_LOG_RATES = 7


def theta_to_phi(theta: np.ndarray) -> np.ndarray:
    """Natural-space ``theta`` -> sampler ``phi`` (first ``N_LOG_RATES`` log)."""
    theta = np.asarray(theta, dtype=float)
    phi = theta.copy()
    phi[..., :N_LOG_RATES] = np.log(theta[..., :N_LOG_RATES])
    return phi


def phi_to_theta(phi: np.ndarray) -> np.ndarray:
    """Sampler ``phi`` -> natural-space ``theta`` (inverse of :func:`theta_to_phi`)."""
    phi = np.asarray(phi, dtype=float)
    theta = phi.copy()
    theta[..., :N_LOG_RATES] = np.exp(phi[..., :N_LOG_RATES])
    return theta


def theta_to_phi_torch(theta):
    """Torch-native counterpart of :func:`theta_to_phi`."""
    import torch
    theta = torch.as_tensor(theta)
    log_part = torch.log(theta[..., :N_LOG_RATES])
    lin_part = theta[..., N_LOG_RATES:]
    return torch.cat([log_part, lin_part], dim=-1)


def phi_to_theta_torch(phi):
    """Torch-native inverse of :func:`theta_to_phi_torch`."""
    import torch
    phi = torch.as_tensor(phi)
    rate_part = torch.exp(phi[..., :N_LOG_RATES])
    lin_part = phi[..., N_LOG_RATES:]
    return torch.cat([rate_part, lin_part], dim=-1)

# ---------------------------------------------------------------------------
# Published prior hyper-parameters
# ---------------------------------------------------------------------------
_LOGN_MU = {
    I_KA2:   -4.2875,
    I_KD:    -3.5090,
    I_KEMPT: -1.9646,
    I_KABS:  -5.4591,
    I_SG:    -3.8000,
}
_LOGN_SIGMA = {
    I_KA2:    0.4274,
    I_KD:     0.6187,
    I_KEMPT:  0.7069,
    I_KABS:   1.4396,
    I_SG:     0.5000,
}

# p2: sqrt(p2) ~ Normal(P2_SQRT_MU, P2_SQRT_SD)
P2_SQRT_MU = 0.11
P2_SQRT_SD = 0.004

# SI: (SI * VG) ~ Gamma(shape=SI_SHAPE, scale=SI_SCALE)
SI_SHAPE = 3.3
SI_SCALE = 5e-4

# Gb: truncated Normal(GB_MU, GB_SD) on [GB_LO, GB_HI]
GB_MU, GB_SD = 119.13, 7.11
GB_LO, GB_HI = 70.0, 180.0

# Soft-constraint weight for the two ordering gates in the MAP residual (large ->
# effectively hard; violations are already rare under these priors).
_ORDER_W = 50.0

# Generous *support* envelope — NOT a prior.
SUPPORT_LO = np.array([1e-3, 1e-3, 2e-2, 1e-4, 1e-3, 1e-5, 5e-3, 70.0])
SUPPORT_HI = np.array([8e-2, 1.5e-1, 8e-1, 1.5e-1, 8e-2, 4e-3, 3e-2, 180.0])


# ---------------------------------------------------------------------------
# Prior centre (published medians) — installed as the population centre
# ---------------------------------------------------------------------------
def prior_center() -> dict:
    """Published prior medians, as a ``THETA_NAMES``-keyed dict."""
    c = {n: 0.0 for n in THETA_NAMES}
    for i in _LOGNORMAL_RATE_IDX:
        c[THETA_NAMES[i]] = float(np.exp(_LOGN_MU[i]))
    c[THETA_NAMES[I_P2]] = float(P2_SQRT_MU ** 2)             # median of sqrt-normal
    # gamma median ~ scale*(shape - 1/3); convert z=SI*VG back to SI
    si_z_med = SI_SCALE * (SI_SHAPE - 1.0 / 3.0)
    c[THETA_NAMES[I_SI]] = float(si_z_med / VG)
    c[THETA_NAMES[I_GB]] = float(GB_MU)
    return c


# ---------------------------------------------------------------------------
# 1) MAP fit support — prior residuals for penalised least-squares
# ---------------------------------------------------------------------------
def prior_residuals(theta: np.ndarray) -> np.ndarray:
    """Prior residual vector ``r`` for one theta, so that ``0.5*sum(r**2)`` equals the
    negative-log-prior (up to a constant).
    """
    theta = np.asarray(theta, dtype=float).ravel()
    out = []

    # Lognormal rates: Normal in phi -> exact Gaussian residual.
    for i in _LOGNORMAL_RATE_IDX:
        phi_i = np.log(max(theta[i], 1e-300))
        out.append((phi_i - _LOGN_MU[i]) / _LOGN_SIGMA[i])

    # p2: Gaussian on sqrt(p2).
    out.append((np.sqrt(max(theta[I_P2], 0.0)) - P2_SQRT_MU) / P2_SQRT_SD)

    # SI: gamma on z = SI*VG, encoded as a single residual r = sqrt(2*Δnll).
    z = max(theta[I_SI] * VG, 1e-300)
    out.append(_gamma_sqrt_residual(z))

    # Gb: Gaussian (truncation handled by the optimiser's box bounds).
    out.append((theta[I_GB] - GB_MU) / GB_SD)

    # Ordering gates as soft hinges (rarely active under these priors).
    out.append(_ORDER_W * max(0.0, np.log(max(theta[I_KA2], 1e-300))
                                   - np.log(max(theta[I_KD], 1e-300))))
    out.append(_ORDER_W * max(0.0, np.log(max(theta[I_KABS], 1e-300))
                                   - np.log(max(theta[I_KEMPT], 1e-300))))
    return np.asarray(out, dtype=float)


def _gamma_sqrt_residual(z: float) -> float:
    """Single residual encoding the Gamma prior on ``z`` as ``sqrt(2*Δnll)``, where Δnll
    is the negative-log-density measured from the gamma mode.
    """
    k, sc = SI_SHAPE, SI_SCALE
    nll = -(k - 1.0) * np.log(z) + z / sc
    z_mode = (k - 1.0) * sc
    nll_min = -(k - 1.0) * np.log(z_mode) + z_mode / sc
    return float(np.sqrt(2.0 * max(0.0, nll - nll_min)))


def order_ok(theta: np.ndarray) -> bool:
    """Whether one theta satisfies both ordering constraints."""
    theta = np.asarray(theta, dtype=float).ravel()
    return bool(theta[I_KA2] < theta[I_KD] and theta[I_KABS] < theta[I_KEMPT])


# ---------------------------------------------------------------------------
# 2) MCMC support — full informative log-prior in phi-space (vectorised)
# ---------------------------------------------------------------------------
def log_prior_phi(phi_batch: np.ndarray) -> np.ndarray:
    """Informative log-prior density in phi-space; ``phi_batch`` is (B, 8)."""
    phi = np.atleast_2d(np.asarray(phi_batch, dtype=float))
    B = phi.shape[0]
    theta = phi.copy()
    theta[:, :N_LOG_RATES] = np.exp(phi[:, :N_LOG_RATES])

    lp = np.zeros(B)

    # Lognormal rates: Normal(mu, sigma) directly on phi.
    for i in _LOGNORMAL_RATE_IDX:
        lp += _norm_logpdf(phi[:, i], _LOGN_MU[i], _LOGN_SIGMA[i])

    # p2: prior on s = sqrt(p2)=exp(phi6/2); +log|ds/dphi6| = log(0.5*s).
    s = np.exp(phi[:, I_P2] / 2.0)
    lp += _norm_logpdf(s, P2_SQRT_MU, P2_SQRT_SD) + np.log(0.5 * s)

    # SI: prior on z = SI*VG = exp(phi5)*VG; +log|dz/dphi5| = log(z).
    z = np.exp(phi[:, I_SI]) * VG
    lp += _gamma_logpdf(z, SI_SHAPE, SI_SCALE) + np.log(z)

    # Gb: truncated Normal on [GB_LO, GB_HI], linear coordinate.
    lp += _truncnorm_logpdf(phi[:, I_GB], GB_MU, GB_SD, GB_LO, GB_HI)

    # Hard gates: support box + ordering.
    in_box = np.all((theta >= SUPPORT_LO) & (theta <= SUPPORT_HI), axis=1)
    order = (theta[:, I_KA2] < theta[:, I_KD]) & (theta[:, I_KABS] < theta[:, I_KEMPT])
    lp = np.where(in_box & order, lp, -np.inf)
    return lp


def _norm_logpdf(x, mu, sd):
    """Log-density of a Normal(mu, sd), evaluated elementwise."""
    return -0.5 * ((x - mu) / sd) ** 2 - np.log(sd) - 0.5 * np.log(2.0 * np.pi)

def _gamma_logpdf(z, k, scale):
    """Log-density of a Gamma(shape=k, scale) on z>0 (-inf for z<=0), via gammaln."""
    z = np.asarray(z, dtype=float)
    out = np.full_like(z, -np.inf)
    pos = z > 0
    out[pos] = ((k - 1.0) * np.log(z[pos]) - z[pos] / scale
                - k * np.log(scale) - gammaln(k))
    return out


def _phi_norm_cdf(x, mu, sd):
    """Standard-normal-based CDF helper, Phi((x-mu)/sd), used to normalise the truncated
    Normal.
    """
    return 0.5 * (1.0 + erf((x - mu) / (sd * np.sqrt(2.0))))


def _truncnorm_logpdf(x, mu, sd, a, b):
    """Log-density of a Normal(mu, sd) truncated to [a, b] (-inf outside), normalised by
    the in-interval mass.
    """
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, -np.inf)
    inside = (x >= a) & (x <= b)
    Z = _phi_norm_cdf(b, mu, sd) - _phi_norm_cdf(a, mu, sd)
    out[inside] = _norm_logpdf(x[inside], mu, sd) - np.log(Z)
    return out


# ---------------------------------------------------------------------------
# 3) SBI support — a torch prior over phi with ordering rejection
# ---------------------------------------------------------------------------
def make_sbi_prior(seed: int | None = None):
    """Return an ``sbi``-compatible prior over phi-space implementing the published
    priors with ordering rejection.
    """
    return _ReplayBGSbiPrior(seed=seed)


def draw_prior_phi(n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``n`` phi-space samples from the published marginals, BEFORE ordering
    rejection.
    """
    phi = np.empty((n, 8))
    for i in _LOGNORMAL_RATE_IDX:
        phi[:, i] = rng.normal(_LOGN_MU[i], _LOGN_SIGMA[i], size=n)
    # p2: draw sqrt(p2) ~ N, reflect to positivity, store log(p2)
    s = np.abs(rng.normal(P2_SQRT_MU, P2_SQRT_SD, size=n))
    phi[:, I_P2] = np.log(np.maximum(s, 1e-6) ** 2)
    # SI: draw z ~ Gamma, store log(SI) = log(z/VG)
    zz = rng.gamma(SI_SHAPE, SI_SCALE, size=n)
    phi[:, I_SI] = np.log(np.maximum(zz, 1e-12) / VG)
    # Gb: truncated normal via rejection
    gb = rng.normal(GB_MU, GB_SD, size=n)
    bad = (gb < GB_LO) | (gb > GB_HI)
    while bad.any():
        gb[bad] = rng.normal(GB_MU, GB_SD, size=int(bad.sum()))
        bad = (gb < GB_LO) | (gb > GB_HI)
    phi[:, I_GB] = gb
    return phi


def sample_prior_phi(n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` phi draws from the published prior WITH the ordering rejection applied."""
    kept, need = [], int(n)
    while need > 0:
        cand = draw_prior_phi(max(need * 2, 64), rng)
        theta = cand.copy()
        theta[:, :N_LOG_RATES] = np.exp(cand[:, :N_LOG_RATES])
        ok = (theta[:, I_KA2] < theta[:, I_KD]) & (theta[:, I_KABS] < theta[:, I_KEMPT])
        kept.append(cand[ok])
        need = n - sum(len(k) for k in kept)
    return np.concatenate(kept, axis=0)[:n]


class _ReplayBGSbiPrior:
    """Minimal torch-facing prior. Kept import-light: torch is imported lazily so this
    module is usable (for the MAP fit / MCMC) without torch installed.
    """

    def __init__(self, seed: int | None = None):
        """Lazily import torch and set up the phi-space prior (event/batch shapes,
        dim=8, seeded RNG) so the module stays importable without torch.
        """
        import torch
        self._torch = torch
        self.event_shape = torch.Size([8])
        self.batch_shape = torch.Size([])
        self.dim = 8
        self._rng = np.random.default_rng(seed)
        # return_type expected by some sbi versions
        self.arg_constraints = {}

    # -- sampling (phi-space) ------------------------------------------------
    def _draw_np(self, n: int) -> np.ndarray:
        """Pre-rejection phi draws. Delegates to the module-level ``draw_prior_phi``."""
        return draw_prior_phi(n, self._rng)

    def sample(self, sample_shape=None):
        """Sample phi from the published prior, rejecting draws that violate ka2<kd or
        kabs<kempt; returns a torch tensor shaped to sample_shape.
        """
        torch = self._torch
        if sample_shape is None or len(tuple(sample_shape)) == 0:
            n = 1
            squeeze = True
        else:
            n = int(np.prod(tuple(sample_shape)))
            squeeze = False
        phi = sample_prior_phi(n, self._rng)
        out = torch.as_tensor(phi, dtype=torch.float32)
        return out[0] if squeeze else out.reshape(*tuple(sample_shape), 8)

    def log_prob(self, phi):
        """Evaluate the published prior log-density at phi (ordering-rejection
        normaliser dropped as a constant); returns a torch tensor.
        """
        torch = self._torch
        arr = np.atleast_2d(np.asarray(phi.detach().cpu().numpy(), dtype=float)) \
            if hasattr(phi, "detach") else np.atleast_2d(np.asarray(phi, dtype=float))
        lp = log_prior_phi(arr)              # ordering-rejection normaliser dropped (constant)
        out = torch.as_tensor(lp, dtype=torch.float32)
        return out if out.ndim else out.reshape(())