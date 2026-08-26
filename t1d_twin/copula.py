"""Multivariate t-copula for representing an MCMC posterior in sampled form."""
from __future__ import annotations

import numpy as np
from scipy.stats import t as _student_t, multivariate_t as _mvt, kendalltau, rankdata

# Small clip so univariate-t ppf/cdf never see exactly 0 or 1 (which map to +-inf).
_U_EPS = 1e-6

# Default degrees-of-freedom search grid.
_DEFAULT_DF_GRID = (3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0)


def _nearest_correlation(A: np.ndarray) -> np.ndarray:
    """Project a symmetric matrix to the nearest valid correlation matrix."""
    B = 0.5 * (A + A.T)
    w, V = np.linalg.eigh(B)
    w = np.clip(w, 1e-8, None)
    C = (V * w) @ V.T
    d = np.sqrt(np.clip(np.diag(C), 1e-12, None))
    C = C / np.outer(d, d)
    np.fill_diagonal(C, 1.0)
    return 0.5 * (C + C.T)


def _empirical_quantile(sorted_col: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Linear-interpolated empirical quantile of ``sorted_col`` at levels ``u``."""
    n = len(sorted_col)
    pos = np.clip(u, 0.0, 1.0) * (n - 1)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    frac = pos - lo
    return sorted_col[lo] * (1.0 - frac) + sorted_col[hi] * frac


def _t_copula_loglik(u: np.ndarray, R: np.ndarray, nu: float) -> float:
    """t-copula log-likelihood on pseudo-observations ``u`` (N,d) given ``R``, ``nu``.
    """
    d = R.shape[0]
    x = _student_t.ppf(np.clip(u, _U_EPS, 1.0 - _U_EPS), df=nu)
    num = _mvt.logpdf(x, loc=np.zeros(d), shape=R, df=nu)          # (N,)
    den = _student_t.logpdf(x, df=nu).sum(axis=1)                  # (N,)
    return float(np.sum(num - den))


class TCopula:
    """Fitted t-copula with empirical margins; draws smoothed posterior realizations."""

    def __init__(self, samples: np.ndarray, corr: np.ndarray, df: float) -> None:
        self.samples = np.asarray(samples, dtype=float)
        self.corr = np.asarray(corr, dtype=float)
        self.df = float(df)
        self.N, self.d = self.samples.shape
        self._sorted = np.sort(self.samples, axis=0)

    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Draw ``n`` realizations (n, d) from the fitted copula + empirical margins."""
        if rng is None:
            rng = np.random.default_rng()
        mv = _mvt(loc=np.zeros(self.d), shape=self.corr, df=self.df,
                  allow_singular=True, seed=rng)
        x = np.atleast_2d(mv.rvs(size=n))
        if x.shape[0] != n:                      # rvs squeezes when n==1
            x = x.reshape(n, self.d)
        u = _student_t.cdf(x, df=self.df)        # (n, d) uniforms
        out = np.empty((n, self.d))
        for j in range(self.d):
            out[:, j] = _empirical_quantile(self._sorted[:, j], u[:, j])
        return out


def fit_t_copula(
    samples: np.ndarray,
    df_grid: tuple[float, ...] = _DEFAULT_DF_GRID,
) -> TCopula:
    """Fit a multivariate t-copula (empirical margins) to posterior ``samples`` (N,d).
    """
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2:
        raise ValueError("samples must be 2-D (N, d)")
    N, d = samples.shape

    # --- correlation via Kendall's tau ---
    R = np.eye(d)
    for i in range(d):
        for j in range(i + 1, d):
            if np.std(samples[:, i]) < 1e-15 or np.std(samples[:, j]) < 1e-15:
                r = 0.0
            else:
                tau, _ = kendalltau(samples[:, i], samples[:, j])
                r = float(np.sin(0.5 * np.pi * tau)) if np.isfinite(tau) else 0.0
            R[i, j] = R[j, i] = r
    R = _nearest_correlation(R)

    # --- pseudo-observations (ranks -> (0,1)) ---
    u = rankdata(samples, axis=0) / (N + 1.0)

    # --- degrees of freedom by grid MLE ---
    best_df, best_ll = df_grid[0], -np.inf
    for nu in df_grid:
        ll = _t_copula_loglik(u, R, nu)
        if np.isfinite(ll) and ll > best_ll:
            best_ll, best_df = ll, nu

    return TCopula(samples, R, best_df)
