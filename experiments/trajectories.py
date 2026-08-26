"""Precompute and ship the benchmark's train/test trajectories."""
from __future__ import annotations

import os
import tempfile
import sys
import json
import time
import hashlib
import argparse
import multiprocessing as mp

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
if os.path.isdir(os.path.join(_root, "t1d_twin")) and _root not in sys.path:
    sys.path.insert(0, _root)

from t1d_twin import value as V                     # noqa: E402
from t1d_twin import units                          # noqa: E402
from experiments import patients as PT              # noqa: E402
from experiments import scenario as SC              # noqa: E402
from experiments import output_paths as OP          # noqa: E402


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


def correction_free_insulin(run, ctrl, dt: float = SC.DT) -> np.ndarray:
    """The insulin a PLANNER could have written down in advance, on the dt grid."""
    df = run.df
    if "insulin_nocorr_mU_kg_min" not in df.columns:
        raise KeyError("run has no `insulin_nocorr_mU_kg_min` column — it was "
                       "produced by an older patients.run_on_patient")
    reps = int(round(float(run.sample_time) / dt))
    return np.repeat(df["insulin_nocorr_mU_kg_min"].to_numpy(), reps)


def _simglucose_version() -> str:
    """Version string for the fingerprint; the meal protocol comes from it."""
    try:
        from importlib.metadata import version
        return str(version("simglucose"))
    except Exception:                                             # noqa: BLE001
        return "unknown"


class StaleTrajectoryError(RuntimeError):
    """A cached .npz was written under a different configuration."""


def config_fingerprint(train_days: int, test_days: int) -> str:
    """Hash of every constant that changes the bytes on disk."""
    cfg = {
        # The meal protocol is simglucose's RandomScenario.
        "meal_source": SC.MEAL_SOURCE,
        "simglucose_version": _simglucose_version(),
        "rel_bolus": SC.REL_BOLUS,
        "basal_factors": SC.BASAL_FACTORS,
        "sample_time": SC.SAMPLE_TIME,
        "sensor": SC.SENSOR_NAME,
        "dt": SC.DT,
        "window_hours": SC.WINDOW_HOURS,
        # The 05:00 anchor and the 29 h run length both change the stored bytes: they
        # move which slice of the day is kept and how much burn-in precedes it.
        "window_start_h": SC.WINDOW_START_H,
        "run_hours": SC.RUN_HOURS,
        "midnight_stride": SC.MIDNIGHT_STRIDE,
        # WHEN the candidate therapy starts.
        "therapy_start_h": SC.THERAPY_START_H,
        # NOT the per-method identification LENGTH: that is a slice taken at scoring
        # time, so it changes results but not the bytes in this file.
        "seed": SC.SEED,
        "train_day0": SC.TRAIN_DAY0,
        "test_day0": SC.TEST_DAY0,
        "train_days": int(train_days),
        "test_days": int(test_days),
        "carb_error": bool(PT.CARB_ERROR_ENABLED),
        # Correction bolus changes the delivered insulin, so it changes every stored
        # byte.
        "correction": bool(PT.CORRECTION_ENABLED),
        # WHICH signal the correction reads ("ig" / "cgm").
        "correction_source": str(PT.CORRECTION_SOURCE),
        "schema": 7,
    }
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def _windowed(run, ctrl, dt: float = SC.DT):
    """Slice the scored 05:00→05:00 window out of one full ``RUN_HOURS`` run."""
    reps = int(round(float(run.sample_time) / dt))
    s1, sr = SC.window_slice(1), SC.window_slice(reps)
    _, insulin_dt, cho_dt = run.replaybg_inputs(dt=dt)
    return {
        "cgm": run.cgm()[s1],
        "ig": run.bg()[s1],
        "insulin_dt": insulin_dt[sr],
        "insulin_nocorr_dt": correction_free_insulin(run, ctrl, dt)[sr],
        "cho_dt": cho_dt[sr],
    }


def build_train(subject, days: int = SC.TRAIN_DAYS, sensor_seed: int = 1) -> dict:
    """``days`` INDEPENDENT runs under the patient's own baseline therapy."""
    ctrl = subject.baseline_controller()
    cgm, ig, ins, ins_nc, cho, meals, trunc = [], [], [], [], [], [], []
    Ib = None
    for day in range(days):
        scen, m = SC.make_scenario(SC.RUN_HOURS, SC.SEED, SC.TRAIN_DAY0 + day)
        run = subject.run(ctrl, scen, SC.RUN_HOURS, sensor_seed=sensor_seed)
        w = _windowed(run, ctrl)
        cgm.append(w["cgm"]); ig.append(w["ig"]); ins.append(w["insulin_dt"])
        ins_nc.append(w["insulin_nocorr_dt"]); cho.append(w["cho_dt"])
        meals.append(len(m))
        trunc.append(bool(run.meta.get("truncated", False)))
        if Ib is None:
            # Basal rate at the window's first sample.
            Ib = float(w["insulin_dt"][0])
    return {
        "cgm": np.asarray(cgm, dtype=np.float32),            # (days, 288)
        "ig": np.asarray(ig, dtype=np.float32),
        "insulin_dt": np.asarray(ins, dtype=np.float32),     # (days, 288*reps)
        # Same covariate a direct multi-horizon method gets at TEST time, so its train
        # and test inputs are drawn from the same distribution.
        "insulin_nocorr_dt": np.asarray(ins_nc, dtype=np.float32),
        "cho_dt": np.asarray(cho, dtype=np.float32),
        "n_meals": np.asarray(meals, dtype=np.int32),
        "BW": float(subject.params["BW"]),
        "sample_time": float(SC.SAMPLE_TIME),
        "dt": float(SC.DT),
        "days": int(days),
        "Ib": float(Ib),
        "truncated": np.asarray(trunc, dtype=bool),
    }


def build_test(subject, days: int = SC.TEST_DAYS, sensor_seed: int = 1) -> dict:
    """``days`` x 20 therapies, each an independent 29 h run from steady state."""
    m = float(subject.dose_mult)
    factors, bnames = SC.relative_factors(m)
    ctrls = subject.therapy_controllers(bolus_factors=factors,
                                        basal_factors=SC.BASAL_FACTORS,
                                        bolus_names=bnames)
    names = SC.therapy_names()
    missing = [n for n in names if n not in ctrls]
    if missing:                                    # ordering is load-bearing
        raise KeyError(f"therapy_names/controllers mismatch: {missing}")

    ig, cgm, ins, ins_nc, cho, rew, trunc = [], [], [], [], [], [], []
    for day in range(days):
        scen, _ = SC.make_scenario(SC.RUN_HOURS, SC.SEED, SC.TEST_DAY0 + day)
        d_ig, d_cgm, d_ins, d_nc, d_cho, d_rew, d_tr = [], [], [], [], [], [], []
        for nm in names:
            run = subject.run(ctrls[nm], scen, SC.RUN_HOURS,
                              sensor_seed=sensor_seed)
            w = _windowed(run, ctrls[nm])
            d_ig.append(w["ig"])
            d_cgm.append(w["cgm"])
            d_ins.append(w["insulin_dt"])
            d_nc.append(w["insulin_nocorr_dt"])
            d_cho.append(w["cho_dt"])
            # Reward is computed on the WINDOW, not the full run.
            d_rew.append(V.reward(w["ig"]))
            d_tr.append(bool(run.meta.get("truncated", False)))
        ig.append(d_ig); cgm.append(d_cgm); ins.append(d_ins); ins_nc.append(d_nc)
        cho.append(d_cho); rew.append(d_rew); trunc.append(d_tr)

    return {
        "therapies": np.asarray(names),
        "ig": np.asarray(ig, dtype=np.float32),            # (D, 20, 288)
        "cgm": np.asarray(cgm, dtype=np.float32),
        "insulin_dt": np.asarray(ins, dtype=np.float32),   # (D, 20, 288*reps)
        "insulin_nocorr_dt": np.asarray(ins_nc, dtype=np.float32),
        "cho_dt": np.asarray(cho, dtype=np.float32),
        "reward": np.asarray(rew, dtype=np.float64),       # (D, 20)
        "truncated": np.asarray(trunc, dtype=bool),
        "days": int(days),
        "sample_time": float(SC.SAMPLE_TIME),
        "dt": float(SC.DT),
    }


def _save(path: str, payload: dict, fingerprint: str, subject) -> None:
    """Write ATOMICALLY: build a temp file, then ``os.replace`` it into position."""
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, f".{os.path.basename(path)}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(tmp, fingerprint=fingerprint, name=subject.name,
                            dose_mult=float(subject.dose_mult),
                            cr_prog=float(subject.cr_prog),
                            cr_true=float(subject.cr_true), **payload)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def load(path: str, expect_fingerprint: str | None = None) -> dict:
    """Load a trajectory npz, refusing a stale one."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} missing — run `python -m experiments.trajectories` first")
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    got = str(d.get("fingerprint", ""))
    if expect_fingerprint is not None and got != expect_fingerprint:
        raise StaleTrajectoryError(
            f"{path} was written under config {got!r} but the current config is "
            f"{expect_fingerprint!r}. Delete it and regenerate — do NOT reuse it.")
    return d


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def _one(task):
    subj, train_days, test_days, force = task
    fp = config_fingerprint(train_days, test_days)
    tr_path = OP.train_trajectory_path(subj.safe_name)
    te_path = OP.test_trajectory_path(subj.safe_name)
    t0 = time.time()
    made = []
    try:
        for path, builder, days, tag in ((tr_path, build_train, train_days, "train"),
                                         (te_path, build_test, test_days, "test")):
            if os.path.exists(path) and not force:
                try:
                    load(path, expect_fingerprint=fp)
                    continue                      # fresh, keep it
                except StaleTrajectoryError:
                    os.remove(path)               # stale, rebuild
            _save(path, builder(subj, days), fp, subj)
            made.append(tag)
        te = load(te_path, expect_fingerprint=fp)
        n_trunc = int(te["truncated"].sum())
        return {"name": subj.name, "made": ",".join(made) or "cached",
                "n_truncated": n_trunc, "secs": round(time.time() - t0, 1)}
    except Exception as e:                                        # noqa: BLE001
        return {"name": subj.name, "error": f"{type(e).__name__}: {e}",
                "secs": round(time.time() - t0, 1)}


class DataDirBusyError(RuntimeError):
    """Another trajectory job is already writing this data date."""


def _lock_path() -> str:
    return os.path.join(OP.DATA_DIR, ".trajectories.lock")


def _acquire_lock(fp: str, note: str):
    """Refuse to run if another job is writing the same ``data/<date>/``."""
    os.makedirs(OP.DATA_DIR, exist_ok=True)
    path = _lock_path()
    payload = (f"pid={os.getpid()} host={os.uname().nodename} "
               f"job={os.environ.get('SLURM_JOB_ID', '-')} fingerprint={fp} "
               f"{note}\n")
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            held = open(path).read().strip()
        except OSError:                                       # noqa: BLE001
            held = "<unreadable>"
        raise DataDirBusyError(
            f"{OP.DATA_DIR} is already being written by: {held}\n"
            f"Two jobs must not generate into the same data date -- that is how "
            f"a mixed-fingerprint data set happens. Wait for it, use a different "
            f"--date, or if you are certain nothing is running, delete "
            f"{path}") from None
    with os.fdopen(fd, "w") as f:
        f.write(payload)
    return path


def main() -> None:
    """CLI: generate and ship the train/test trajectory files."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kinds", default=",".join(PT.DEFAULT_KINDS),
                    help="comma-separated simglucose cohorts: adult, adolescent, "
                         "child. Default: all three (30 subjects).")
    ap.add_argument("--train-days", type=int, default=SC.TRAIN_DAYS)
    ap.add_argument("--test-days", type=int, default=SC.TEST_DAYS)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if a fresh cached file exists")
    args = ap.parse_args()

    from experiments.logsetup import quiet as _quiet, is_verbose
    _quiet(is_verbose())

    kinds = tuple(k.strip() for k in args.kinds.split(",") if k.strip())
    subjects = PT.benchmark_subjects(kinds)
    print(f"[traj] {len(subjects)} built-in simglucose subjects ({', '.join(kinds)})")
    if args.limit:
        subjects = subjects[:args.limit]

    fp = config_fingerprint(args.train_days, args.test_days)
    print(f"[traj] {len(subjects)} patients, {args.train_days} train days, "
          f"{args.test_days} test days x {SC.N_THERAPIES} therapies, config {fp}")
    print(f"[traj] train -> {os.path.dirname(OP.train_trajectory_path('x'))}")
    print(f"[traj] test  -> {os.path.dirname(OP.test_trajectory_path('x'))}")

    lock = _acquire_lock(fp, f"train_days={args.train_days} "
                             f"test_days={args.test_days} kinds={args.kinds}")
    tasks = [(s, args.train_days, args.test_days, args.force) for s in subjects]
    t0 = time.time()
    rows = []
    try:
        if args.jobs > 1:
            _warm_font_cache()
            with mp.get_context("spawn").Pool(args.jobs) as pool:
                for i, r in enumerate(pool.imap_unordered(_one, tasks), 1):
                    rows.append(r)
                    if i % 10 == 0 or i == len(tasks):
                        print(f"  [{i}/{len(tasks)}] {time.time() - t0:.0f}s",
                              flush=True)
        else:
            for i, t in enumerate(tasks, 1):
                rows.append(_one(t))
                print(f"  [{i}/{len(tasks)}] {rows[-1]}", flush=True)
    finally:
        # Released even on crash or SIGTERM-during-cleanup, so a failed run does not
        # wedge the date for the next attempt.
        try:
            os.remove(lock)
        except OSError:                                        # noqa: BLE001
            pass

    bad = [r for r in rows if r.get("error")]
    trunc = [r for r in rows if r.get("n_truncated")]
    print(f"[traj] {len(rows) - len(bad)} ok, {len(bad)} failed "
          f"({time.time() - t0:.0f}s)")
    for r in bad[:10]:
        print(f"    FAIL {r['name']}: {r['error']}")
    if trunc:
        # A truncated run means simglucose ended the episode early (BG left its safe
        # range) and run_on_patient padded it at the terminal danger value.
        n = sum(r["n_truncated"] for r in trunc)
        print(f"[traj] NOTE {n} (day, therapy) cells across {len(trunc)} patients "
              f"terminated early and were padded at the terminal glucose")
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
