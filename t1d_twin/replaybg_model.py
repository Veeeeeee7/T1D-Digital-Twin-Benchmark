"""ReplayBG forward model (Cappon et al., 2023)."""
from __future__ import annotations

import numpy as np

# --- fixed population parameters (Cappon et al., 2023) --- (based on replaybg population, not ours)
VI = 0.126     # insulin distribution volume [l/kg]
KE = 0.127     # insulin fractional clearance [1/min]
BETA = 8.0     # insulin appearance delay [min]
F = 0.9        # fraction of intestinal glucose absorbed [-]
VG = 1.45      # glucose distribution volume [dl/kg]
ALPHA = 7.0    # plasma->interstitium delay [min]

# --- rho(G) hypoglycemia function (eq. 5) ---
GTH = 60.0     # hypoglycemic threshold [mg/dL]
R1 = 1.44
R2 = 0.81

# --- free parameter vector layout ---
THETA_NAMES = ["ka2", "kd", "kempt", "kabs", "SG", "SI", "p2", "Gb"]


def theta_to_array(theta) -> np.ndarray:
    """Accept a dict or array-like and return an (B, 8) float array."""
    if isinstance(theta, dict):
        return np.array([theta[n] for n in THETA_NAMES], dtype=float)
    return np.asarray(theta, dtype=float)


def rho(G: np.ndarray, Gb: np.ndarray) -> np.ndarray:
    """Hypoglycemia insulin-action amplification (eq. 5), vectorised."""
    G = np.asarray(G, dtype=float)
    Gb = np.asarray(Gb, dtype=float)
    Gs = np.maximum(G, 1.0)                       # guard log of small/neg G
    lnG, lnGb, lnGth = np.log(Gs), np.log(Gb), np.log(GTH)
    mid = 1.0 + 10.0 * R1 * (lnG ** R2 - lnGb ** R2) ** 2
    low = 1.0 + 10.0 * R1 * (lnGth ** R2 - lnGb ** R2) ** 2
    return np.where(G >= Gb, 1.0, np.where(G <= GTH, low, mid))


def steady_state(theta: np.ndarray, Ib: float) -> np.ndarray:
    """Steady-state initial condition for constant basal insulin ``Ib`` and no meal."""
    theta = np.atleast_2d(theta_to_array(theta))
    ka2, kd, _, _, _, _, _, Gb = theta.T
    Isc1 = Ib / (VI * kd)
    Isc2 = Ib / (VI * ka2)
    Ip = np.full_like(Gb, Ib / (VI * KE))          # = Ipb
    zeros = np.zeros_like(Gb)
    x0 = np.stack([Isc1, Isc2, Ip, zeros, zeros, zeros, Gb, zeros, Gb], axis=1)
    return x0 if x0.shape[0] > 1 else x0[0]


def _deriv(x: np.ndarray, I_del: float, cho: float,
           theta: np.ndarray, Ipb: float) -> np.ndarray:
    """State derivative. ``x`` is (B, 9); ``theta`` is (B, 8); inputs scalar."""
    Isc1, Isc2, Ip, Qsto1, Qsto2, Qgut, G, X, IG = x.T
    ka2, kd, kempt, kabs, SG, SI, p2, Gb = theta.T

    dIsc1 = -kd * Isc1 + I_del / VI
    dIsc2 = kd * Isc1 - ka2 * Isc2
    dIp = ka2 * Isc2 - KE * Ip

    dQsto1 = -kempt * Qsto1 + cho
    dQsto2 = kempt * Qsto1 - kempt * Qsto2
    dQgut = kempt * Qsto2 - kabs * Qgut
    Ra = F * kabs * Qgut

    dG = -(SG + rho(G, Gb) * X) * G + SG * Gb + Ra / VG
    dX = -p2 * (X - SI * (Ip - Ipb))
    dIG = -(IG - G) / ALPHA

    return np.stack([dIsc1, dIsc2, dIp, dQsto1, dQsto2, dQgut, dG, dX, dIG], axis=1)


def _delayed_insulin(insulin: np.ndarray, Ib: float, dt: float) -> np.ndarray:
    """Shift insulin by the appearance delay BETA; pre-window value is basal Ib."""
    k = int(round(BETA / dt))
    out = np.empty_like(insulin)
    out[:k] = Ib
    out[k:] = insulin[:len(insulin) - k]
    return out


def simulate(theta, insulin: np.ndarray, cho: np.ndarray, Ib: float,
             dt: float = 1.0, x0: np.ndarray | None = None,
             return_states: bool = False):
    """Integrate the model with fixed-step RK4."""
    theta_arr = np.atleast_2d(theta_to_array(theta))
    B = theta_arr.shape[0]
    insulin = np.asarray(insulin, dtype=float)
    cho = np.asarray(cho, dtype=float)
    T = insulin.shape[0]

    Ipb = Ib / (VI * KE)
    if x0 is None:
        x = np.atleast_2d(steady_state(theta_arr, Ib)).copy()
    else:
        x = np.atleast_2d(np.array(x0, float)).copy()
    I_del = _delayed_insulin(insulin, Ib, dt)

    ig = np.empty((B, T))
    states = np.empty((B, T, 9)) if return_states else None
    for m in range(T):
        Id, ch = I_del[m], cho[m]
        k1 = _deriv(x, Id, ch, theta_arr, Ipb)
        k2 = _deriv(x + 0.5 * dt * k1, Id, ch, theta_arr, Ipb)
        k3 = _deriv(x + 0.5 * dt * k2, Id, ch, theta_arr, Ipb)
        k4 = _deriv(x + dt * k3, Id, ch, theta_arr, Ipb)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        ig[:, m] = x[:, 8]
        if return_states:
            states[:, m, :] = x

    t = (np.arange(T) + 1) * dt
    single = np.ndim(theta_to_array(theta)) == 1
    ig_out = ig[0] if single else ig
    if return_states:
        st = states[0] if single else states
        return t, ig_out, st
    return t, ig_out


def sample_indices(n_samples: int, sample_time: float, dt: float = 1.0) -> np.ndarray:
    """Samples indices into the dt-grid IG that align with observation samples."""
    reps = sample_time / dt
    if abs(reps - round(reps)) > 1e-9:
        raise ValueError("sample_time must be an integer multiple of dt")
    reps = int(round(reps))
    return (np.arange(n_samples) + 1) * reps - 1