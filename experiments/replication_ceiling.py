"""max_spearman — the score a PERFECT twin gets, and a determinism check."""
from __future__ import annotations

import os
import sys
import csv
import time
import argparse
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

#: Sensor seed the shipped test set was generated under (``trajectories.build_test``
#: default).
REFERENCE_SEED = 1

#: Replicate realizations per patient.
DEFAULT_SEEDS = 4


def replicate_rewards(subject, days: int, sensor_seed: int):
    """Re-run every (day, therapy) cell at ``sensor_seed``."""
    out = TRAJ.build_test(subject, days=days, sensor_seed=sensor_seed)
    return (np.asarray(out["reward"], float), np.asarray(out["ig"], float))


def _kind(name: str) -> str:
    return str(name).split("#")[0]


def _one(task):
    subj, days_req, seeds, fp, verify = task
    t0 = time.time()
    try:
        test = TRAJ.load(OP.test_trajectory_path(subj.safe_name),
                         expect_fingerprint=fp)
    except Exception as e:                                        # noqa: BLE001
        return {"name": subj.name, "error": f"{type(e).__name__}: {e}"}, []

    try:
        therapies = [str(t) for t in test["therapies"]]
        true_reward = np.asarray(test["reward"], float)            # (D, P)
        true_ig = np.asarray(test["ig"], float)                    # (D, P, T)
        D = true_reward.shape[0] if not days_req else min(days_req,
                                                          true_reward.shape[0])
        true_reward, true_ig = true_reward[:D], true_ig[:D]

        # Cheap determinism guard: re-running the REFERENCE seed must reproduce the
        # shipped bytes.
        drift = float("nan")
        if verify:
            _, ig1 = replicate_rewards(subj, 1, REFERENCE_SEED)
            # Compare in float32, the dtype the .npz stores.
            drift = float(np.max(np.abs(ig1[0].astype(np.float32)
                                        - np.asarray(test["ig"], np.float32)[0])))

        rows = []
        for s in range(REFERENCE_SEED + 1, REFERENCE_SEED + 1 + seeds):
            rep_reward, rep_ig = replicate_rewards(subj, D, s)
            for d in range(D):
                tr, pr = true_reward[d], rep_reward[d]
                rows.append({
                    "patient": subj.name, "kind": _kind(subj.name),
                    "day": d, "seed": s,
                    "spearman": SCORE.spearman(tr, pr),
                    "regret": SCORE.regret(tr, pr),
                    "regret_norm": SCORE.regret_norm(tr, pr),
                    "rmse": float(np.mean([SCORE.rmse(true_ig[d, j], rep_ig[d, j])
                                           for j in range(len(therapies))])),
                    "mard": float(np.mean([SCORE.mard(true_ig[d, j], rep_ig[d, j])
                                           for j in range(len(therapies))])),
                    "true_best": therapies[int(np.argmax(tr))],
                    "rep_best": therapies[int(np.argmax(pr))],
                    "reward_spread": float(np.max(tr) - np.min(tr)),
                })

        sp = np.array([r["spearman"] for r in rows], float)
        # Reward SD across realizations, per (day, therapy), then averaged: the
        # magnitude of the noise the ranking is fighting.
        adj = [float(np.mean(np.abs(np.diff(np.sort(true_reward[d])))))
               for d in range(D)]

        summary = {
            "name": subj.name,
            "kind": _kind(subj.name),
            "n_days": D,
            "n_seeds": seeds,
            "n_pairs": len(rows),
            "max_spearman": float(np.nanmean(sp)),
            "max_spearman_sd": float(np.nanstd(sp, ddof=1)) if len(sp) > 1 else 0.0,
            "max_spearman_min": float(np.nanmin(sp)),
            "max_spearman_p10": float(np.nanpercentile(sp, 10)),
            "regret_floor": float(np.mean([r["regret"] for r in rows])),
            "regret_norm_floor": float(np.mean([r["regret_norm"] for r in rows])),
            "regret_norm_floor_max": float(np.max([r["regret_norm"] for r in rows])),
            "rmse_floor": float(np.mean([r["rmse"] for r in rows])),
            "mard_floor": float(np.mean([r["mard"] for r in rows])),
            "reward_spread": float(np.mean([r["reward_spread"] for r in rows])),
            "adj_gap_mean": float(np.mean(adj)),
            "n_degenerate": int(np.sum(~np.isfinite(sp))),
            "ref_drift": drift,
            "secs": round(time.time() - t0, 1),
            "error": "",
        }
        return summary, rows
    except Exception as e:                                        # noqa: BLE001
        return ({"name": subj.name, "error": f"{type(e).__name__}: {e}",
                 "secs": round(time.time() - t0, 1)}, [])


SUMMARY_COLUMNS = ("name", "kind", "n_days", "n_seeds", "n_pairs",
                   "max_spearman", "max_spearman_sd",
                   "max_spearman_min", "max_spearman_p10",
                   "regret_floor", "regret_norm_floor", "regret_norm_floor_max",
                   "rmse_floor", "mard_floor",
                   "reward_spread", "adj_gap_mean", "n_degenerate",
                   "ref_drift", "secs", "error")

DAY_COLUMNS = ("patient", "kind", "day", "seed", "spearman", "regret",
               "regret_norm", "rmse", "mard",
               "true_best", "rep_best", "reward_spread")


def _write(path: str, rows, columns) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(columns))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in columns})
    return path


def main() -> None:
    """CLI: measure the per-patient replication ceiling."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kinds", default=",".join(PT.DEFAULT_KINDS))
    ap.add_argument("--seeds", type=int, default=DEFAULT_SEEDS,
                    help="replicate sensor realizations per patient. Cost is "
                         "linear in this; the reference is the shipped set, so "
                         "N seeds give N x days comparisons.")
    ap.add_argument("--days", type=int, default=0,
                    help="0 = every shipped test day. Cost is linear in this too, "
                         "and days are the cheaper axis to cut: seeds set how many "
                         "comparisons each day contributes.")
    ap.add_argument("--train-days", type=int, default=SC.TRAIN_DAYS,
                    help="only used to reconstruct the trajectory fingerprint")
    ap.add_argument("--test-days", type=int, default=SC.TEST_DAYS,
                    help="only used to reconstruct the trajectory fingerprint")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--verify", action="store_true",
                    help="re-run the REFERENCE seed on day 0 and report the max "
                         "|IG| drift from the shipped bytes. Should be exactly "
                         "0.0; anything else means the plant is nondeterministic "
                         "given its seed and this whole measurement is void.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from experiments.logsetup import quiet as _quiet, is_verbose
    _quiet(is_verbose())

    kinds = tuple(k.strip() for k in args.kinds.split(",") if k.strip())
    subjects = PT.benchmark_subjects(kinds)
    if args.limit:
        subjects = subjects[:args.limit]

    fp = TRAJ.config_fingerprint(args.train_days, args.test_days)
    out_path = args.out or OP.replication_ceiling_path()
    day_path = OP.replication_ceiling_days_path()

    print(f"[ceiling] {len(subjects)} subjects ({', '.join(kinds)}) x "
          f"{args.days or SC.TEST_DAYS} days x {SC.N_THERAPIES} therapies x "
          f"{args.seeds} replicate seeds, config {fp}")
    print(f"[ceiling] reference = the SHIPPED test set (sensor_seed="
          f"{REFERENCE_SEED}); replicates are seeds "
          f"{REFERENCE_SEED + 1}..{REFERENCE_SEED + args.seeds}")

    tasks = [(s, args.days, args.seeds, fp, args.verify) for s in subjects]
    t0 = time.time()
    summaries, day_rows = [], []
    if args.jobs > 1 and len(tasks) > 1:
        with mp.get_context("fork").Pool(min(args.jobs, len(tasks))) as pool:
            for i, (s, rows) in enumerate(pool.imap_unordered(_one, tasks), 1):
                summaries.append(s)
                day_rows.extend(rows)
                if i % 5 == 0 or i == len(tasks):
                    print(f"  [{i}/{len(tasks)}] {time.time() - t0:.0f}s", flush=True)
    else:
        for i, t in enumerate(tasks, 1):
            s, rows = _one(t)
            summaries.append(s)
            day_rows.extend(rows)
            print(f"  [{i}/{len(tasks)}] {s.get('name')} "
                  f"{s.get('max_spearman', s.get('error'))} "
                  f"({s.get('secs')}s)", flush=True)

    order = {s.name: i for i, s in enumerate(subjects)}
    summaries.sort(key=lambda r: order.get(r["name"], 1 << 30))
    _write(out_path, summaries, SUMMARY_COLUMNS)
    _write(day_path, day_rows, DAY_COLUMNS)

    good = [r for r in summaries if not r.get("error")]
    bad = [r for r in summaries if r.get("error")]
    print(f"[ceiling] wrote {out_path}  ({len(good)} ok, {len(bad)} failed, "
          f"{time.time() - t0:.0f}s)")
    print(f"[ceiling] wrote {day_path}  ({len(day_rows)} rows)")
    for r in bad[:10]:
        print(f"    FAIL {r['name']}: {r['error']}")

    if not good:
        raise SystemExit(1)

    drift = [r["ref_drift"] for r in good if np.isfinite(r.get("ref_drift", np.nan))]
    if drift:
        mx = max(drift)
        verdict = ("OK — bit-identical" if mx == 0.0 else
                   "NONZERO — the plant is not deterministic given its seed, "
                   "so this whole measurement is void")
        print(f"[ceiling] reference re-run drift: max |IG| = {mx:.3e} mg/dL "
              f"({verdict})")

    sp = np.array([r["max_spearman"] for r in good])
    rn = np.array([r["regret_norm_floor"] for r in good])
    rm = np.array([r["rmse_floor"] for r in good])
    md = np.array([r["mard_floor"] for r in good])
    print(f"\n  max_spearman       n={len(good)}   mean {sp.mean():.6f}   "
          f"min {sp.min():.6f}")
    print(f"  regret_floor       mean {np.mean([r['regret_floor'] for r in good]):.4f}")
    print(f"  regret_norm_floor  mean {rn.mean():.6f}   max {rn.max():.6f}")
    print(f"  rmse_floor         mean {rm.mean():.4f} mg/dL   "
          f"mard_floor {md.mean():.4f}%")

    # ---- verdict -----------------------------------------------------------
    # With CORRECTION_SOURCE = "ig" the plant is deterministic given its seed, so
    # every quantity above has a KNOWN exact value.
    exact = PT.CORRECTION_SOURCE == "ig" or not PT.CORRECTION_ENABLED
    if not exact:
        print(f"\n  NOTE correction_source={PT.CORRECTION_SOURCE!r}: the plant is "
              f"sensor-driven, so these are measurements, not exact values.")
        return
    checks = [("max_spearman == 1", float(sp.min()) == 1.0),
              ("regret_norm_floor == 0", float(rn.max()) == 0.0),
              ("rmse_floor == 0", float(rm.max()) == 0.0),
              ("mard_floor == 0", float(md.max()) == 0.0)]
    if drift:
        checks.append(("reference re-run drift == 0", max(drift) == 0.0))
    print(f"\n  determinism check (correction_source={PT.CORRECTION_SOURCE!r}):")
    for name, ok in checks:
        print(f"    {'PASS' if ok else '**FAIL**'}  {name}")
    if all(ok for _, ok in checks):
        print("  VERDICT: PASS — the plant is deterministic, the true ranking is "
              "a property of the patient and meals alone.")
    else:
        print("  VERDICT: FAIL — something other than the sensor is stochastic. "
              "Every benchmark number on this vintage is suspect.")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
