"""Scoring harness for the counterfactual therapy-ranking benchmark."""
from __future__ import annotations

import os
import csv
import numpy as np
from scipy.stats import spearmanr, rankdata

#: Column order for both tables.
LONG_COLUMNS = ("patient", "day", "therapy", "method",
                "true_reward", "pred_reward", "true_rank", "pred_rank",
                "rmse", "mard")
DECISION_COLUMNS = ("patient", "day", "method",
                    "spearman", "regret", "regret_norm",
                    "rmse", "mard", "true_best", "pred_best",
                    "n_therapies", "degenerate", "nonfinite")

DECISION_COLS = ["spearman", "regret", "regret_norm"]
FIDELITY_COLS = ["rmse", "mard"]


def ranks(rewards: np.ndarray) -> np.ndarray:
    """Rank 1 = best (highest reward); ties share the average rank."""
    r = np.asarray(rewards, dtype=float)
    return rankdata(-r, method="average")


def spearman(true_r: np.ndarray, pred_r: np.ndarray) -> float:
    """Spearman rank correlation; NaN if either side is constant or non-finite."""
    t = np.asarray(true_r, float)
    p = np.asarray(pred_r, float)
    if t.size < 2 or not np.isfinite(p).all() or not np.isfinite(t).all():
        return float("nan")
    if np.ptp(t) == 0 or np.ptp(p) == 0:
        return float("nan")
    return float(spearmanr(t, p).correlation)


def regret(true_r: np.ndarray, pred_r: np.ndarray) -> float:
    """True reward lost by taking the twin's top pick. >= 0; 0 is optimal."""
    t = np.asarray(true_r, float)
    p = np.asarray(pred_r, float)
    if not np.isfinite(p).all():
        return float("nan")
    return float(np.max(t) - t[int(np.argmax(p))])


def regret_norm(true_r: np.ndarray, pred_r: np.ndarray) -> float:
    """``regret`` divided by the patient-day's own reward spread -> [0, 1]."""
    t = np.asarray(true_r, float)
    p = np.asarray(pred_r, float)
    if not np.isfinite(p).all():
        return float("nan")
    spread = float(np.max(t) - np.min(t))
    if spread <= 0.0:
        return 0.0
    return float(regret(t, p) / spread)


def rmse(true_ig: np.ndarray, pred_ig: np.ndarray) -> float:
    """Root-mean-square error between true and predicted IG [mg/dL]."""
    t = np.asarray(true_ig, float).ravel()
    p = np.asarray(pred_ig, float).ravel()
    return float(np.sqrt(np.mean((p - t) ** 2)))


def mard(true_ig: np.ndarray, pred_ig: np.ndarray) -> float:
    """Mean absolute relative difference [%], true IG as reference."""
    t = np.asarray(true_ig, float).ravel()
    p = np.asarray(pred_ig, float).ravel()
    return float(100.0 * np.mean(np.abs(p - t) / np.clip(np.abs(t), 1.0, None)))


def score_patient(patient: str, method: str, therapies,
                  true_reward: np.ndarray, pred_reward: np.ndarray,
                  true_ig: np.ndarray | None = None,
                  pred_ig: np.ndarray | None = None):
    """Score one (patient, method) over all test days."""
    true_reward = np.asarray(true_reward, float)
    pred_reward = np.asarray(pred_reward, float)
    if true_reward.shape != pred_reward.shape:
        raise ValueError(f"reward shape mismatch: {true_reward.shape} vs "
                         f"{pred_reward.shape}")
    D, P = true_reward.shape
    if len(therapies) != P:
        raise ValueError(f"{len(therapies)} therapy names for {P} columns")

    long_rows, dec_rows = [], []
    for d in range(D):
        tr, pr = true_reward[d], pred_reward[d]
        t_rank, p_rank = ranks(tr), ranks(pr)
        per_rmse = np.full(P, np.nan)
        per_mard = np.full(P, np.nan)
        if true_ig is not None and pred_ig is not None:
            for j in range(P):
                per_rmse[j] = rmse(true_ig[d, j], pred_ig[d, j])
                per_mard[j] = mard(true_ig[d, j], pred_ig[d, j])
        for j, nm in enumerate(therapies):
            long_rows.append({
                "patient": patient, "day": d, "therapy": str(nm), "method": method,
                "true_reward": float(tr[j]), "pred_reward": float(pr[j]),
                "true_rank": float(t_rank[j]), "pred_rank": float(p_rank[j]),
                "rmse": float(per_rmse[j]), "mard": float(per_mard[j]),
            })
        sp = spearman(tr, pr)
        dec_rows.append({
            "patient": patient, "day": d, "method": method,
            "spearman": sp,
            "regret": regret(tr, pr),
            "regret_norm": regret_norm(tr, pr),
            "rmse": float(np.mean(per_rmse)) if np.isfinite(per_rmse).any() else float("nan"),
            "mard": float(np.mean(per_mard)) if np.isfinite(per_mard).any() else float("nan"),
            "true_best": str(therapies[int(np.argmax(tr))]),
            "pred_best": str(therapies[int(np.argmax(pr))]),
            "n_therapies": P,
            # A twin predicting one reward for every therapy has made no decision.
            "degenerate": int(np.ptp(pr) == 0),
            # Count of cells whose predicted reward was NaN or infinite.
            "nonfinite": int((~np.isfinite(pr)).sum()),
        })
    return long_rows, dec_rows


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------
def _write(path: str, rows, columns) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(columns))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in columns})
    return path


def write_long(path: str, rows) -> str:
    """Write the per-therapy long table to ``path``."""
    return _write(path, rows, LONG_COLUMNS)


def write_decisions(path: str, rows) -> str:
    """Write the per-patient-day decision table to ``path``."""
    return _write(path, rows, DECISION_COLUMNS)
