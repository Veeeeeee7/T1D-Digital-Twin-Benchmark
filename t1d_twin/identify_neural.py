"""Neural impulse-response twin — the benchmark's NONLINEAR black-box arm."""
from __future__ import annotations

from typing import Optional

import numpy as np

from .identify_linear import (
    L_CHO_DEFAULT, L_INS_DEFAULT, bin_to_sample_grid, design_matrix,
)
from .simglucose_adapter import RunResult
from .twin import Twin

#: Hidden layer widths.
HIDDEN_DEFAULT: tuple[int, ...] = (64, 64)

#: Adam settings.
LR_DEFAULT: float = 1e-3
WEIGHT_DECAY_DEFAULT: float = 1e-4
BATCH_DEFAULT: int = 256

#: Epoch cap and early-stopping patience, both measured against the HELD-OUT day
#: split.
MAX_EPOCHS: int = 200
PATIENCE: int = 20

#: Fraction of training DAYS held out for early stopping (never a random sample split
#: — see the module docstring).
VAL_DAY_FRACTION: float = 0.2

#: Torch intra-op threads during a fit.
TORCH_THREADS: int = 1

#: Training days consumed — every shipped day, matching ``linear`` so the two black-
#: box arms differ only in hypothesis class.
FIT_DAYS_DEFAULT: int = 0        # 0 = all shipped days


# ---------------------------------------------------------------------------
# Standardisation — an MLP needs it, the ridge did not
# ---------------------------------------------------------------------------
class _Scaler:
    """Column-wise standardiser with a zero-variance guard."""

    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean = np.asarray(mean, dtype=float)
        self.std = np.asarray(std, dtype=float)

    @classmethod
    def fit(cls, X: np.ndarray) -> "_Scaler":
        """Fit per-column mean and std, guarding near-constant columns."""
        X = np.asarray(X, dtype=float)
        std = X.std(axis=0)
        std[std < 1e-8] = 1.0
        return cls(X.mean(axis=0), std)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=float) - self.mean) / self.std


def _build_mlp(n_in: int, hidden: tuple[int, ...]):
    import torch.nn as nn
    layers, prev = [], int(n_in)
    for h in hidden:
        layers += [nn.Linear(prev, int(h)), nn.ReLU()]
        prev = int(h)
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


def fit_mlp(X: np.ndarray, y: np.ndarray, day_id: np.ndarray,
            hidden: tuple[int, ...] = HIDDEN_DEFAULT,
            lr: float = LR_DEFAULT, weight_decay: float = WEIGHT_DECAY_DEFAULT,
            batch: int = BATCH_DEFAULT, max_epochs: int = MAX_EPOCHS,
            patience: int = PATIENCE,
            val_day_fraction: float = VAL_DAY_FRACTION,
            seed: int = 42, verbose: bool = False):
    """Train the MLP with early stopping on a held-out DAY split."""
    import torch

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    day_id = np.asarray(day_id)
    if len(X) != len(y) or len(X) != len(day_id):
        raise ValueError(f"ragged inputs: {len(X)} rows, {len(y)} targets, "
                         f"{len(day_id)} day labels")

    days = np.unique(day_id)
    n_val = int(np.floor(len(days) * float(val_day_fraction)))
    held_out = n_val >= 1 and len(days) - n_val >= 1
    if held_out:
        val_days = set(days[-n_val:].tolist())     # LAST days, not random ones
        val_mask = np.array([d in val_days for d in day_id])
    else:
        val_mask = np.zeros(len(X), dtype=bool)
    tr_mask = ~val_mask

    scaler = _Scaler.fit(X[tr_mask])
    y_mean = float(y[tr_mask].mean())
    y_std = float(y[tr_mask].std())
    if y_std < 1e-8:
        y_std = 1.0

    prev_threads = torch.get_num_threads()
    torch.set_num_threads(int(TORCH_THREADS))
    try:
        torch.manual_seed(int(seed))
        Xtr = torch.as_tensor(scaler(X[tr_mask]), dtype=torch.float32)
        ytr = torch.as_tensor((y[tr_mask] - y_mean) / y_std,
                              dtype=torch.float32).unsqueeze(1)
        if held_out:
            Xva = torch.as_tensor(scaler(X[val_mask]), dtype=torch.float32)
            yva = torch.as_tensor((y[val_mask] - y_mean) / y_std,
                                  dtype=torch.float32).unsqueeze(1)

        model = _build_mlp(X.shape[1], hidden)
        opt = torch.optim.Adam(model.parameters(), lr=float(lr),
                               weight_decay=float(weight_decay))
        loss_fn = torch.nn.MSELoss()
        gen = torch.Generator().manual_seed(int(seed))

        best = float("inf")
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        bad_epochs = 0
        n = Xtr.shape[0]
        for epoch in range(int(max_epochs)):
            model.train()
            perm = torch.randperm(n, generator=gen)
            for i in range(0, n, int(batch)):
                idx = perm[i: i + int(batch)]
                opt.zero_grad()
                loss_fn(model(Xtr[idx]), ytr[idx]).backward()
                opt.step()

            model.eval()
            with torch.no_grad():
                cur = (float(loss_fn(model(Xva), yva)) if held_out
                       else float(loss_fn(model(Xtr), ytr)))
            if cur < best - 1e-6:
                best = cur
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if held_out and bad_epochs >= int(patience):
                    if verbose:
                        print(f"  [neural] early stop at epoch {epoch}")
                    break

        model.load_state_dict(best_state)
        model.eval()
    finally:
        torch.set_num_threads(prev_threads)

    return model, scaler, y_mean, y_std, float(np.sqrt(best) * y_std), held_out


class NeuralTwin(Twin):
    """Point-estimate twin over a fitted MLP on lagged covariates."""

    def __init__(self, model, scaler: _Scaler, y_mean: float, y_std: float,
                 Ib: float, sample_time: float, dt: float = 1.0,
                 l_ins: int = L_INS_DEFAULT, l_cho: int = L_CHO_DEFAULT,
                 sensor_name: str = "GuardianRT",
                 hidden: tuple[int, ...] = HIDDEN_DEFAULT,
                 val_rmse: float | None = None,
                 held_out: bool = True) -> None:
        self.model = model
        self.scaler = scaler
        self.y_mean = float(y_mean)
        self.y_std = float(y_std)
        self.Ib = float(Ib)
        self.sample_time = float(sample_time)
        self.dt = float(dt)
        self.l_ins = int(l_ins)
        self.l_cho = int(l_cho)
        self.sensor_name = sensor_name
        self.hidden = tuple(int(h) for h in hidden)
        self.val_rmse = val_rmse
        self.held_out = bool(held_out)

    # ------------------------------------------------------------------
    # Twin ABC
    # ------------------------------------------------------------------
    def predict_ig(self, insulin: np.ndarray, cho: np.ndarray,
                   n_samples: Optional[int] = None):
        """Predicted IG on the **dt grid**, as the ABC requires."""
        import torch

        reps = int(round(self.sample_time / self.dt))
        ins5 = bin_to_sample_grid(insulin, reps)
        cho5 = bin_to_sample_grid(cho, reps)
        X = design_matrix(ins5, cho5, self.Ib, self.l_ins, self.l_cho)
        with torch.no_grad():
            z = self.model(torch.as_tensor(self.scaler(X), dtype=torch.float32))
        pred5 = z.squeeze(1).numpy().astype(float) * self.y_std + self.y_mean
        ig = np.repeat(pred5, reps)
        if len(ig) < len(insulin):      # trailing partial bin, benchmark-impossible
            ig = np.concatenate([ig, np.full(len(insulin) - len(ig), ig[-1])])
        return ig, np.tile(ig, (2, 1))

    def summary(self) -> dict:
        """Architecture and fit diagnostics. No physiological parameters exist."""
        n_par = sum(int(p.numel()) for p in self.model.parameters())
        out = {
            "n_parameters": float(n_par),
            "n_hidden_layers": float(len(self.hidden)),
            "l_ins": float(self.l_ins),
            "l_cho": float(self.l_cho),
        }
        if self.val_rmse is not None:
            out["val_rmse" if self.held_out else "insample_rmse"] = float(self.val_rmse)
        res = {k: {"median": float(v), "ci95": (float(v), float(v))}
               for k, v in out.items()}
        res["_meta"] = {"held_out_validation": self.held_out,
                        "hidden": list(self.hidden)}
        return res


# ---------------------------------------------------------------------------
# Glue
# ---------------------------------------------------------------------------
def identify_twin_from_runs(runs, dt: float = 1.0,
                            l_ins: int = L_INS_DEFAULT,
                            l_cho: int = L_CHO_DEFAULT,
                            hidden: tuple[int, ...] = HIDDEN_DEFAULT,
                            sensor_name: str = "GuardianRT",
                            seed: int = 42,
                            verbose: bool = False) -> NeuralTwin:
    """Identify a :class:`NeuralTwin` from one or more runs (each one day)."""
    runs = [runs] if isinstance(runs, RunResult) else list(runs)
    if not runs:
        raise ValueError("need at least one run to identify a NeuralTwin")

    Xs, ys, ds = [], [], []
    Ib = None
    for i, run in enumerate(runs):
        _, insulin, cho = run.replaybg_inputs(dt=dt)
        if Ib is None:
            Ib = float(insulin[0])
        reps = int(round(float(run.sample_time) / dt))
        Xi = design_matrix(bin_to_sample_grid(insulin, reps),
                           bin_to_sample_grid(cho, reps), Ib, l_ins, l_cho)
        yi = np.asarray(run.cgm(), dtype=float)
        if len(Xi) != len(yi):
            raise ValueError(f"day {i}: {len(Xi)} covariate rows vs "
                             f"{len(yi)} CGM samples")
        Xs.append(Xi)
        ys.append(yi)
        ds.append(np.full(len(yi), i))

    X, y, day_id = np.vstack(Xs), np.concatenate(ys), np.concatenate(ds)
    # ``held_out`` comes back FROM the fit rather than being re-derived here.
    model, scaler, y_mean, y_std, val_rmse, held_out = fit_mlp(
        X, y, day_id, hidden=hidden, seed=seed, verbose=verbose)
    return NeuralTwin(model, scaler, y_mean, y_std, Ib,
                      float(runs[0].sample_time), dt, l_ins, l_cho, sensor_name,
                      hidden=hidden, val_rmse=val_rmse, held_out=held_out)


def identify_twin_from_run(run: RunResult, dt: float = 1.0, **kw) -> NeuralTwin:
    """Single-run convenience wrapper, matching the other methods' entry point."""
    return identify_twin_from_runs([run], dt=dt, **kw)
