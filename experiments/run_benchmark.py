"""The benchmark — counterfactual therapy ranking, one row per
patient/day/therapy/method.
"""
from __future__ import annotations

import os
import tempfile
import sys
import time
import argparse
import traceback
import multiprocessing as mp

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
if os.path.isdir(os.path.join(_root, "t1d_twin")) and _root not in sys.path:
    sys.path.insert(0, _root)

from t1d_twin import value as V                     # noqa: E402
from experiments import patients as PT              # noqa: E402
from experiments import scenario as SC              # noqa: E402
from experiments import scoring as SCORE            # noqa: E402
from experiments import trajectories as TRAJ        # noqa: E402
from experiments import output_paths as OP          # noqa: E402

#: The arms that can enter the benchmark.
ALL_METHODS = ("mcmc", "sbi", "linear", "prior", "neural", "ukf")
DEFAULT_METHODS = ("mcmc", "sbi", "linear", "prior", "neural", "ukf")

#: Which insulin covariate each method is fed.
METHOD_COVARIATES = {
    "mcmc": "insulin_dt",
    "sbi": "insulin_dt",
    # ``linear`` replays a therapy's recorded covariates through a fitted impulse
    # response.
    "linear": "insulin_dt",
    # ``prior`` is the mechanistic replay path with a prior ensemble in place of a
    # posterior, so it consumes exactly what ``mcmc`` consumes.
    "prior": "insulin_dt",
    # ``neural`` is a one-pass replay model over a whole therapy's recorded
    # covariates, NOT the shelved autoregressive twin.
    "neural": "insulin_dt",
    # ``ukf`` filters the ReplayBG ODE forward over the recorded inputs and then
    # replays therapies through the same path ``mcmc`` uses.
    "ukf": "insulin_dt",
    # "direct": "insulin_nocorr_dt", # re-enable with the shelved sequence twin
}
DEFAULT_COVARIATE = "insulin_dt"

#: How much identification history each method consumes, in hours.
METHOD_FIT_HOURS = {
    "mcmc": SC.PAPER_FIT_HOURS,      # 22 h
    "sbi": SC.PAPER_FIT_HOURS,       # 22 h
    "linear": SC.WINDOW_HOURS,       # 24 h per day, x METHOD_FIT_DAYS
    "prior": SC.PAPER_FIT_HOURS,     # 22 h — inert, the arm reads no data
    "neural": SC.WINDOW_HOURS,       # 24 h per day, x METHOD_FIT_DAYS
    "ukf": SC.PAPER_FIT_HOURS,       # 22 h — matched to mcmc, see below
}
DEFAULT_FIT_HOURS = SC.WINDOW_HOURS  # 24 h — the whole 05:00→05:00 window

#: How many INDEPENDENT training days a method consumes.
ALL_TRAINING_DAYS = 0
METHOD_FIT_DAYS = {
    "linear": ALL_TRAINING_DAYS,
    # ``neural`` matches ``linear`` exactly.
    "neural": ALL_TRAINING_DAYS,
}
DEFAULT_FIT_DAYS = 1


def _warm_font_cache() -> None:
    """Build matplotlib's font cache in the parent before any worker spawns.

    simglucose imports matplotlib, and a ``spawn`` pool re-imports it in every
    child; a cold cache then has N workers building it at once in one
    MPLCONFIGDIR, which can deadlock. Touching it here makes it warm for them.
    """
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault(
        "MPLCONFIGDIR",
        os.path.join(tempfile.gettempdir(),
                     f"mpl_{os.environ.get('USER', 'u')}_"
                     f"{os.environ.get('SLURM_JOB_ID', os.getpid())}"))
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        from matplotlib import font_manager
        font_manager.fontManager.findfont("DejaVu Sans")
    except Exception:
        pass


def resolve_fit_days(meth: str, n_shipped: int) -> int:
    """Days this method consumes, clamped to what the trajectory file carries."""
    spec = METHOD_FIT_DAYS.get(meth, DEFAULT_FIT_DAYS)
    return int(n_shipped) if spec <= 0 else min(int(spec), int(n_shipped))

#: WHICH training day the identification window is taken from.
FIT_WINDOW_DAY = 0


class _ArrayRun:
    """Minimal ``RunResult`` stand-in over stored arrays."""

    def __init__(self, cgm, ig, insulin_dt, cho_dt, sample_time, dt,
                 name="", BW=1.0):  # noqa: D107
        self._cgm = np.asarray(cgm, float)
        self._ig = np.asarray(ig, float)
        self._insulin = np.asarray(insulin_dt, float)
        self._cho = np.asarray(cho_dt, float)
        self.sample_time = float(sample_time)
        self.dt = float(dt)
        self.patient_name = name
        self.BW = float(BW)
        self.sensor_name = SC.SENSOR_NAME
        self.sensor_seed = 1
        self.meta = {}

    def cgm(self):
        """Noisy CGM over the scored window [mg/dL]."""
        return self._cgm

    def bg(self):
        """Noise-free interstitial glucose over the scored window [mg/dL]."""
        return self._ig

    def replaybg_inputs(self, dt: float = 1.0):
        """Stored insulin and carbohydrate covariates on the ODE grid."""
        if abs(dt - self.dt) > 1e-9:
            raise ValueError(f"stored covariates are on a dt={self.dt} grid, "
                             f"asked for dt={dt}")
        return np.arange(len(self._insulin)) * self.dt, self._insulin, self._cho


def fit_window(train: dict, day: int = FIT_WINDOW_DAY,
               hours: float = DEFAULT_FIT_HOURS,
               insulin_key: str = DEFAULT_COVARIATE) -> _ArrayRun:
    """Take the first ``hours`` of one training day's 05:00→05:00 window."""
    st = float(train["sample_time"])
    dt = float(train["dt"])
    reps = int(round(st / dt))
    cgm = np.asarray(train["cgm"])
    if cgm.ndim != 2:
        raise ValueError(
            f"training trajectory has shape {cgm.shape}; expected (days, "
            f"{SC.N_OBS}) — regenerate it.")
    if not 0 <= day < cgm.shape[0]:
        raise ValueError(f"fit day {day} outside the {cgm.shape[0]} shipped days")
    n = int(round(hours * 60.0 / st))
    if n > cgm.shape[1]:
        raise ValueError(
            f"fit window of {hours} h = {n} samples exceeds the shipped "
            f"{cgm.shape[1]}-sample ({SC.WINDOW_HOURS} h) window")
    return _ArrayRun(
        cgm=cgm[day, :n],
        ig=np.asarray(train["ig"])[day, :n],
        insulin_dt=np.asarray(train[insulin_key])[day, :n * reps],
        cho_dt=np.asarray(train["cho_dt"])[day, :n * reps],
        sample_time=st, dt=dt, name=str(train.get("name", "")),
        BW=float(train["BW"]),
    )


def _predict_all(twin, test: dict, insulin_key: str = DEFAULT_COVARIATE) -> np.ndarray:
    """(D, P, T) predicted noise-free IG for every test cell."""
    if insulin_key not in test:
        raise KeyError(
            f"{insulin_key!r} not in the shipped test set (has: "
            f"{sorted(k for k in test if k.startswith('insulin'))}). Regenerate "
            f"the trajectories — this data predates the covariate split.")
    D, P, _ = test["ig"].shape
    out = np.empty_like(np.asarray(test["ig"], dtype=float))
    for d in range(D):
        for j in range(P):
            out[d, j] = twin.generate_cgm(
                test[insulin_key][d, j].astype(float),
                test["cho_dt"][d, j].astype(float),
                add_noise=False,
            )
    return out


def _rewards_from_ig(ig: np.ndarray) -> np.ndarray:
    D, P, _ = ig.shape
    return np.array([[V.reward(ig[d, j]) for j in range(P)] for d in range(D)])


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------
def _fit_mcmc(run, seed: int, cfg: dict):
    from t1d_twin.identify_mcmc import identify_twin_from_run
    return identify_twin_from_run(run, dt=run.dt, seed=seed,
                                  nburn=cfg["mcmc_burn"], nsample=cfg["mcmc_sample"])


def _fit_sbi(run, seed: int, cfg: dict):
    from t1d_twin.identify_sbi import identify_twin_from_run
    return identify_twin_from_run(run, dt=run.dt, seed=seed,
                                  n_train=cfg["sbi_train"], infer_ic=True)


def _fit_linear(runs, seed: int, cfg: dict):
    """``runs`` is a LIST — this is the one arm that consumes several days."""
    from t1d_twin.identify_linear import identify_twin_from_runs
    return identify_twin_from_runs(runs, dt=runs[0].dt, seed=seed,
                                   sensor_name=SC.SENSOR_NAME)


def _fit_prior(run, seed: int, cfg: dict):
    """The no-personalization baseline. ``run`` supplies ``Ib`` and nothing else."""
    from t1d_twin.identify_prior import identify_twin_from_run
    return identify_twin_from_run(run, dt=run.dt, seed=seed,
                                  sensor_name=SC.SENSOR_NAME)


def _fit_ukf(run, seed: int, cfg: dict):
    """Recursive filter identification — one forward pass over the window."""
    from t1d_twin.identify_ukf import identify_twin_from_run
    return identify_twin_from_run(run, dt=run.dt, seed=seed,
                                  sensor_name=SC.SENSOR_NAME)


def _fit_neural(runs, seed: int, cfg: dict):
    """``runs`` is a LIST — same multi-day budget as ``linear``, deliberately."""
    from t1d_twin.identify_neural import identify_twin_from_runs
    return identify_twin_from_runs(runs, dt=runs[0].dt, seed=seed,
                                   sensor_name=SC.SENSOR_NAME)


# ---------------------------------------------------------------------------
# Per-patient work
# ---------------------------------------------------------------------------
def _one(task):
    name, safe, methods, seed, cfg = task
    t0 = time.time()
    long_rows, dec_rows, timings, errors = [], [], {}, {}
    try:
        fp = TRAJ.config_fingerprint(cfg["train_days"], cfg["test_days"])
        train = TRAJ.load(OP.train_trajectory_path(safe), expect_fingerprint=fp)
        test = TRAJ.load(OP.test_trajectory_path(safe), expect_fingerprint=fp)
        therapies = [str(t) for t in test["therapies"]]
        true_reward = np.asarray(test["reward"], float)
        true_ig = np.asarray(test["ig"], float)
    except Exception as e:                                        # noqa: BLE001
        return {"name": name, "error": f"{type(e).__name__}: {e}",
                "long": [], "dec": [], "timings": {}, "secs": 0.0}

    for meth in methods:
        m0 = time.time()
        try:
            # Cut the window per method: BOTH its length (the method's own published
            # protocol) and its covariate variant.
            key = METHOD_COVARIATES.get(meth, DEFAULT_COVARIATE)
            hrs = METHOD_FIT_HOURS.get(meth, DEFAULT_FIT_HOURS)
            ndays = resolve_fit_days(meth, np.asarray(train["cgm"]).shape[0])
            runs = [fit_window(train, day=FIT_WINDOW_DAY + i, hours=hrs,
                               insulin_key=key) for i in range(ndays)]
            run = runs[0]
            if meth == "mcmc":
                twin = _fit_mcmc(run, seed, cfg)
            elif meth == "sbi":
                twin = _fit_sbi(run, seed, cfg)
            elif meth == "linear":
                twin = _fit_linear(runs, seed, cfg)
            elif meth == "prior":
                twin = _fit_prior(run, seed, cfg)
            elif meth == "neural":
                twin = _fit_neural(runs, seed, cfg)
            elif meth == "ukf":
                twin = _fit_ukf(run, seed, cfg)
            else:
                raise ValueError(f"unknown method {meth!r}")
            pred_ig = _predict_all(twin, test, key)
            lr, dr = SCORE.score_patient(name, meth, therapies,
                                         true_reward, _rewards_from_ig(pred_ig),
                                         true_ig=true_ig, pred_ig=pred_ig)
            long_rows += lr
            dec_rows += dr
        except Exception as e:                                    # noqa: BLE001
            errors[meth] = f"{type(e).__name__}: {e}"
            if cfg.get("traceback"):
                errors[meth] += "\n" + traceback.format_exc()
        timings[meth] = round(time.time() - m0, 1)

    return {"name": name, "long": long_rows, "dec": dec_rows,
            "timings": timings, "errors": errors,
            "secs": round(time.time() - t0, 1)}


def _progress(i: int, n: int, r: dict) -> str:
    """One progress line. Surfaces BOTH failure channels."""
    line = f"  [{i}/{n}] {r['name']} {r['secs']}s {r.get('timings', {})}"
    if r.get("error"):
        line += f"  LOAD-FAILED {r['error']}"
    if r.get("errors"):
        line += f"  ERRORS {r['errors']}"
    return line


def main() -> None:
    """CLI: fit every method on every patient, predict, and score."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kinds", default=",".join(PT.DEFAULT_KINDS),
                    help="comma-separated simglucose cohorts: adult, adolescent, "
                         "child. Default: all three (30 subjects).")
    ap.add_argument("--methods", default=",".join(DEFAULT_METHODS),
                    help=f"comma-separated subset of {ALL_METHODS}")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-days", type=int, default=SC.TRAIN_DAYS)
    ap.add_argument("--test-days", type=int, default=SC.TEST_DAYS)
    ap.add_argument("--mcmc-burn", type=int, default=None)
    ap.add_argument("--mcmc-sample", type=int, default=None)
    ap.add_argument("--sbi-train", type=int, default=None)
    ap.add_argument("--traceback", action="store_true")
    args = ap.parse_args()

    from experiments.logsetup import quiet as _quiet, is_verbose
    _quiet(is_verbose())

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    bad = [m for m in methods if m not in ALL_METHODS]
    if bad:
        raise SystemExit(f"unknown methods {bad}; choose from {list(ALL_METHODS)}")

    # Resolve each method's published defaults LAZILY.
    def _mcmc_defaults():
        from t1d_twin import identify_mcmc as IM
        return IM.DEFAULT_BURN, IM.DEFAULT_NSAMPLE

    def _sbi_default():
        from t1d_twin import identify_sbi as ISB
        return ISB.N_TRAIN_DEFAULT

    burn, nsamp = ((args.mcmc_burn, args.mcmc_sample)
                   if (args.mcmc_burn and args.mcmc_sample)
                   else _mcmc_defaults() if "mcmc" in methods else (0, 0))
    cfg = {
        "train_days": args.train_days, "test_days": args.test_days,
        "mcmc_burn": args.mcmc_burn or burn,
        "mcmc_sample": args.mcmc_sample or nsamp,
        "sbi_train": args.sbi_train or (_sbi_default() if "sbi" in methods else 0),
        "traceback": args.traceback,
    }

    kinds = tuple(k.strip() for k in args.kinds.split(",") if k.strip())
    subjects = PT.benchmark_subjects(kinds)
    print(f"[bench] {len(subjects)} built-in simglucose subjects ({', '.join(kinds)})")
    if args.limit:
        subjects = subjects[:args.limit]

    print(f"[bench] {len(subjects)} patients x {len(methods)} methods "
          f"{methods} x {args.test_days} days x {SC.N_THERAPIES} therapies")
    print(f"[bench] fit window = training day {FIT_WINDOW_DAY} "
          f"({SC.WINDOW_START_H:04.1f}h -> {SC.WINDOW_START_H:04.1f}h next day); "
          f"length is per method:")
    for m in methods:
        h = METHOD_FIT_HOURS.get(m, DEFAULT_FIT_HOURS)
        n = int(round(h * 60 / SC.SAMPLE_TIME))
        d = resolve_fit_days(m, args.train_days)
        print(f"    {m:10s} {d:2d} x {h:5.1f} h = {n * d:5d} samples at "
              f"{SC.SAMPLE_TIME:.0f} min, covariate "
              f"{METHOD_COVARIATES.get(m, DEFAULT_COVARIATE)}")
    print(f"[bench] mcmc burn/sample = {cfg['mcmc_burn']}/{cfg['mcmc_sample']}, "
          f"sbi n_train = {cfg['sbi_train']}")

    tasks = [(s.name, s.safe_name, methods, args.seed, cfg) for s in subjects]
    t0 = time.time()
    results = []
    if args.jobs > 1:
        _warm_font_cache()
        with mp.get_context("spawn").Pool(args.jobs) as pool:
            for i, r in enumerate(pool.imap_unordered(_one, tasks), 1):
                results.append(r)
                print(_progress(i, len(tasks), r), flush=True)
    else:
        for i, t in enumerate(tasks, 1):
            results.append(_one(t))
            r = results[-1]
            print(_progress(i, len(tasks), r), flush=True)

    long_rows = [r for res in results for r in res["long"]]
    dec_rows = [r for res in results for r in res["dec"]]
    if not dec_rows:
        raise SystemExit("[bench] no results produced — every patient failed")

    lp = SCORE.write_long(OP.long_table_path(), long_rows)
    dp = SCORE.write_decisions(OP.decisions_table_path(), dec_rows)
    print(f"\n[bench] {len(long_rows)} long rows -> {lp}")
    print(f"[bench] {len(dec_rows)} decision rows -> {dp}")

    failed = [r for r in results if r.get("error")]
    errs = {}
    for r in results:
        for m, e in (r.get("errors") or {}).items():
            errs.setdefault(m, []).append((r["name"], e))
    if failed:
        print(f"[bench] {len(failed)} patients failed to load:")
        for r in failed[:5]:
            print(f"    {r['name']}: {r['error']}")
    for m, lst in errs.items():
        print(f"[bench] method {m} failed on {len(lst)} patients, e.g. "
              f"{lst[0][0]}: {lst[0][1].splitlines()[0]}")

    # Inline preview so a run is readable without the analysis step.
    print(f"\n[bench] {time.time() - t0:.0f}s total. Per-method means "
          f"(paired analysis is in analyze_benchmark):")
    print(f"  {'method':10s} {'spearman':>9s} {'regret_norm':>12s} "
          f"{'rmse':>8s} {'mard':>8s} {'degen':>6s} {'nonfin':>7s}")
    for m in methods:
        rows = [r for r in dec_rows if r["method"] == m]
        if not rows:
            continue
        sp = np.array([r["spearman"] for r in rows], float)
        # nanmean: regret_norm is NaN on any patient-day with non-finite predictions.
        print(f"  {m:10s} {np.nanmean(sp):9.3f} "
              f"{np.nanmean([r['regret_norm'] for r in rows]):12.3f} "
              f"{np.nanmean([r['rmse'] for r in rows]):8.1f} "
              f"{np.nanmean([r['mard'] for r in rows]):8.1f} "
              f"{sum(r['degenerate'] for r in rows):6d} "
              f"{sum(r.get('nonfinite', 0) for r in rows):7d}")
    print(f"\n[bench] next: python -m experiments.analyze_benchmark")
    if failed or errs:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
