"""Leave-one-patient-out and leave-one-kind-out sensitivity of the paper's headline claims."""
from __future__ import annotations

import os
import sys
import csv
import argparse
import itertools
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
if _root not in sys.path:
    sys.path.insert(0, _root)

from experiments.analyze_benchmark import (      # noqa: E402
    LOWER_IS_BETTER, METHOD_ORDER, PRETTY_METHOD, _friedman, _matrix, _read,
    holm_pairwise, per_patient,
)

#: Metrics whose pairwise verdicts are tracked (the paper's post-hoc table).
PAIR_METRICS = ("rmse", "mard", "spearman", "regret_norm")

#: Metrics whose Friedman test the paper reports.
FRIEDMAN_METRICS = ("rmse", "mard", "spearman", "regret", "regret_norm")

ALPHA = 0.05

#: Smallest cohort ``holm_pairwise`` and ``_friedman`` define a statistic on.
MIN_PATIENTS = 3


# ---------------------------------------------------------------------------
# One fold
# ---------------------------------------------------------------------------
def drop(pp: dict, patients) -> dict:
    """``pp`` without ``patients``."""
    out = set(patients)
    return {m: {p: v for p, v in d.items() if p not in out} for m, d in pp.items()}


def cohort_means(pp: dict, methods, metric: str) -> dict:
    """``method -> mean over patients`` of the per-patient means."""
    return {m: float(np.nanmean([pp[m][p][metric] for p in pp[m]])) for m in methods}


def cohort_ranks(means: dict, metric: str) -> dict:
    """``method -> rank`` of the cohort means, 1 = best; ties share the lower rank, NaN unranked."""
    sign = 1.0 if metric in LOWER_IS_BETTER else -1.0
    key = {m: sign * v for m, v in means.items() if np.isfinite(v)}
    return {m: 1 + sum(k < key[m] for k in key.values()) if m in key else float("nan")
            for m in means}


def pair_stats(pp: dict, methods, metric: str, friedman_p: float) -> dict:
    """``(a, b) -> {delta, raw, holm, sep}`` per pair; ``sep`` is Friedman-gated like the report."""
    gate = bool(np.isfinite(friedman_p) and friedman_p < ALPHA)
    used, idx_pairs, deltas, raw, holm = holm_pairwise(pp, metric, methods)
    got = {}
    for (i, j), d, r, h in zip(idx_pairs, deltas, raw, holm):
        got[(used[i], used[j])] = {"delta": d, "raw": r, "holm": float(h),
                                   "sep": bool(gate and np.isfinite(h) and h < ALPHA)}
    nan = float("nan")
    return {pair: got.get(pair, {"delta": nan, "raw": nan, "holm": nan, "sep": False})
            for pair in itertools.combinations(methods, 2)}


def fold_summary(pp: dict, methods) -> dict:
    """Every statistic a claim can read, on one cohort."""
    s = {"n": len(set.intersection(*[set(pp[m]) for m in methods])),
         "means": {}, "ranks": {}, "pairs": {}, "friedman": {}}
    for metric in FRIEDMAN_METRICS:
        s["means"][metric] = cohort_means(pp, methods, metric)
        s["ranks"][metric] = cohort_ranks(s["means"][metric], metric)
        s["friedman"][metric] = _friedman(_matrix(pp, methods, metric)[1])
    for metric in PAIR_METRICS:
        s["pairs"][metric] = pair_stats(pp, methods, metric, s["friedman"][metric])
    s["neural_vs_zero"] = _one_sample_p(pp, "neural", "spearman")
    return s


def _one_sample_p(pp: dict, method: str, metric: str) -> float:
    """Two-sided Wilcoxon of ``method``'s per-patient ``metric`` against 0."""
    if method not in pp:
        return float("nan")
    x = np.array([pp[method][p][metric] for p in pp[method]], float)
    x = x[np.isfinite(x)]
    if len(x) < 3 or not np.any(x != 0):
        return float("nan")
    try:
        return float(wilcoxon(x).pvalue)
    except ValueError:
        return float("nan")


def _pair(s, metric, a, b):
    """Pair stats regardless of the order the pair was stored in; delta is a − b."""
    p = s["pairs"][metric]
    if (a, b) in p:
        return p[(a, b)]
    q = dict(p[(b, a)])
    q["delta"] = -q["delta"]
    return q


# ---------------------------------------------------------------------------
# The paper's claims, each a predicate on a fold summary
# ---------------------------------------------------------------------------
def _sep(s, metric, a, b):
    """Whether ``a`` and ``b`` are separated on ``metric`` in fold summary ``s``."""
    return _pair(s, metric, a, b)["sep"]


def _spearman_order_under_rmse(s, methods):
    """Cohort Spearman ranks, listed in cohort RMSE order."""
    by_rmse = sorted(methods, key=lambda m: s["ranks"]["rmse"][m])
    return tuple(s["ranks"]["spearman"][m] for m in by_rmse)


CLAIMS = [
    ("C1", "MCMC vs SBI: separated on RMSE, not on Spearman",
     lambda s, M: _sep(s, "rmse", "mcmc", "sbi") and not _sep(s, "spearman", "mcmc", "sbi")),
    ("C2", "Prior: worst cohort RMSE and second-best cohort Spearman",
     lambda s, M: s["ranks"]["rmse"]["prior"] == len(M) and s["ranks"]["spearman"]["prior"] == 2),
    ("C3", "SBI vs Prior: separated on RMSE, not on Spearman",
     lambda s, M: _sep(s, "rmse", "sbi", "prior") and not _sep(s, "spearman", "sbi", "prior")),
    ("C4", "Neural vs Prior: separated on both, Neural better on RMSE and worse on Spearman",
     lambda s, M: (_sep(s, "rmse", "neural", "prior") and _sep(s, "spearman", "neural", "prior")
                   and _pair(s, "rmse", "neural", "prior")["delta"] < 0
                   and _pair(s, "spearman", "neural", "prior")["delta"] < 0)),
    ("C5", "UKF vs Neural: not separated on RMSE, separated on Spearman",
     lambda s, M: not _sep(s, "rmse", "ukf", "neural") and _sep(s, "spearman", "ukf", "neural")),
    ("C6", "UKF vs Linear: not separated on RMSE, separated on Spearman",
     lambda s, M: not _sep(s, "rmse", "ukf", "linear") and _sep(s, "spearman", "ukf", "linear")),
    ("C7", "Neural vs Linear: separated on RMSE, Neural better",
     lambda s, M: _sep(s, "rmse", "neural", "linear") and _pair(s, "rmse", "neural", "linear")["delta"] < 0),
    ("C8", "MCMC ranks first on both cohort RMSE and cohort Spearman",
     lambda s, M: s["ranks"]["rmse"]["mcmc"] == 1 and s["ranks"]["spearman"]["mcmc"] == 1),
    ("C9", "Spearman ranks in RMSE order are exactly 1, 3, 5, 4, 6, 2 (Table 3)",
     lambda s, M: _spearman_order_under_rmse(s, M) == (1, 3, 5, 4, 6, 2)),
    ("C10", "Friedman rejects on all five metrics",
     lambda s, M: all(np.isfinite(p) and p < ALPHA for p in s["friedman"].values())),
]


# ---------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------
def kind_of(patient: str) -> str:
    """``adult#001`` -> ``adult``."""
    return patient.split("#")[0]


def _all_folds(pp: dict, methods, by: str):
    """``(patients, [(label, patients_to_drop)])`` before the size check."""
    patients = sorted(set.intersection(*[set(pp[m]) for m in methods]))
    if by == "patient":
        return patients, [(p, [p]) for p in patients]
    if by == "kind":
        groups = defaultdict(list)
        for p in patients:
            groups[kind_of(p)].append(p)
        return patients, [(k, v) for k, v in sorted(groups.items())]
    raise ValueError(by)


def folds(pp: dict, methods, by: str):
    """``[(label, patients_to_drop)]`` by ``patient`` or ``kind``, omitting folds below ``MIN_PATIENTS``."""
    patients, fs = _all_folds(pp, methods, by)
    return [(lab, d) for lab, d in fs if len(patients) - len(d) >= MIN_PATIENTS]


def skipped_folds(pp: dict, methods, by: str) -> list:
    """Labels of the folds ``folds`` omits for leaving too few patients."""
    patients, fs = _all_folds(pp, methods, by)
    return [lab for lab, d in fs if len(patients) - len(d) < MIN_PATIENTS]


def run(pp: dict, methods, by: str):
    """``(full_summary, [(label, summary)])``."""
    full = fold_summary(pp, methods)
    out = [(label, fold_summary(drop(pp, d), methods)) for label, d in folds(pp, methods, by)]
    return full, out


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def pair_rows(full, fs, methods):
    """Long rows: one per (fold, metric, pair)."""
    rows = []
    for label, s in [("<none>", full)] + fs:
        for metric in PAIR_METRICS:
            for (a, b), st in s["pairs"][metric].items():
                f = full["pairs"][metric][(a, b)]
                rows.append({"dropped": label, "n": s["n"], "metric": metric,
                             "a": a, "b": b, "delta": st["delta"], "raw_p": st["raw"],
                             "holm_p": st["holm"], "separated": int(st["sep"]),
                             "agrees_with_full": int(st["sep"] == f["sep"])})
    return rows


def claim_rows(full, fs, methods):
    """Rows: one per (fold, claim)."""
    rows = []
    for label, s in [("<none>", full)] + fs:
        for cid, text, fn in CLAIMS:
            rows.append({"dropped": label, "n": s["n"], "claim": cid,
                         "holds": int(bool(fn(s, methods)))})
    return rows


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _p(x):
    if not np.isfinite(x):
        return "n/a"
    return "<1e-4" if x < 1e-4 else f"{x:.4f}"


def _span(xs):
    """``(min, max, n_nonfinite)`` over the finite values; NaN ends when none are."""
    x = np.asarray(xs, float)
    ok = x[np.isfinite(x)]
    if not len(ok):
        return float("nan"), float("nan"), len(x)
    return float(ok.min()), float(ok.max()), int(len(x) - len(ok))


def _na(k):
    return f" ({k} n/a)" if k else ""


def _skipped(labels):
    if not labels:
        return ""
    return f"; skipped {', '.join(labels)} (would leave < {MIN_PATIENTS} patients)"


def report(full, by_patient, by_kind, methods, src, skipped=None) -> list[str]:
    """Markdown report; ``skipped`` maps ``patient``/``kind`` to omitted fold labels."""
    P = PRETTY_METHOD
    skipped = skipped or {}
    L = ["# Leave-one-out sensitivity of the headline claims", "",
         f"source: `{src}`", "",
         f"- full cohort: n = {full['n']} patients, k = {len(methods)} methods",
         f"- leave-one-patient-out: {len(by_patient)} folds of n = "
         f"{by_patient[0][1]['n'] if by_patient else 'n/a'}"
         + _skipped(skipped.get("patient")),
         f"- leave-one-kind-out: {len(by_kind)} folds "
         "(" + ", ".join("%s n=%d" % (k, s["n"]) for k, s in by_kind) + ")"
         + _skipped(skipped.get("kind")),
         "- every fold recomputes Friedman, Holm-corrected pairwise Wilcoxon "
         f"(Holm over all {len(methods) * (len(methods) - 1) // 2} pairs) and "
         "cohort-mean ranks with `analyze_benchmark`'s own helpers",
         "- a pair counts as separated only where that metric's Friedman also "
         "rejects, as `analyze_benchmark` runs its post-hoc tables only then", "",
         "## 1. Claims", "",
         "| claim | full cohort | holds, leave-one-patient-out | fails when dropping | "
         "holds, leave-one-kind-out | fails when dropping |",
         "|---|:--:|:--:|---|:--:|---|"]
    for cid, text, fn in CLAIMS:
        fh = fn(full, methods)
        fp = [lab for lab, s in by_patient if not fn(s, methods)]
        fk = [lab for lab, s in by_kind if not fn(s, methods)]
        L.append(f"| **{cid}** {text} | {'yes' if fh else 'NO'} | "
                 f"{len(by_patient) - len(fp)}/{len(by_patient)} | "
                 f"{', '.join(fp) or '—'} | {len(by_kind) - len(fk)}/{len(by_kind)} | "
                 f"{', '.join(fk) or '—'} |")
    L += ["", "## 2. Holm p range across leave-one-patient-out folds", "",
          "Every pair on every tracked metric. `agree` counts the folds whose "
          "separated / not-separated verdict matches the full cohort's.", ""]
    for metric in PAIR_METRICS:
        L += [f"### {metric}", "",
              "| A vs B | full Holm p | min | max | agree | verdict |",
              "|---|---:|---:|---:|:--:|:--:|"]
        for (a, b), f in full["pairs"][metric].items():
            lo, hi, na = _span([s["pairs"][metric][(a, b)]["holm"] for _, s in by_patient])
            ag = sum(s["pairs"][metric][(a, b)]["sep"] == f["sep"] for _, s in by_patient)
            L.append(f"| {P[a]} vs {P[b]} | {_p(f['holm'])} | {_p(lo)} | "
                     f"{_p(hi)}{_na(na)} | {ag}/{len(by_patient)} | "
                     f"{'separated' if f['sep'] else 'not separated'} |")
        L.append("")
    L += ["## 3. Cohort means across leave-one-patient-out folds", "",
          "| method | RMSE full | RMSE min–max | Spearman full | Spearman min–max |",
          "|---|---:|---:|---:|---:|"]
    for m in methods:
        r0, r1, rn = _span([s["means"]["rmse"][m] for _, s in by_patient])
        s0, s1, sn = _span([s["means"]["spearman"][m] for _, s in by_patient])
        L.append(f"| {P[m]} | {full['means']['rmse'][m]:.1f} | {r0:.1f}–{r1:.1f}{_na(rn)} | "
                 f"{full['means']['spearman'][m]:.3f} | {s0:.3f}–{s1:.3f}{_na(sn)} |")
    z0, z1, zn = _span([s["neural_vs_zero"] for _, s in by_patient])
    L += ["", "## 4. Diagnostic (not a paper claim)", "",
          f"Neural's per-patient Spearman vs 0, two-sided one-sample Wilcoxon: "
          f"full p = {_p(full['neural_vs_zero'])}, leave-one-patient-out range "
          f"{_p(z0)}–{_p(z1)}{_na(zn)}.", ""]
    return L


def main() -> None:
    """CLI: read the decisions table, write the sensitivity CSVs and report."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--decisions", default=None)
    ap.add_argument("--out-dir", default=None,
                    help="defaults to the decisions table's directory")
    args = ap.parse_args()

    if args.decisions is None:
        from experiments import output_paths as OP
        args.decisions = OP.decisions_table_path()
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.decisions))

    dec = _read(args.decisions)
    if not dec:
        raise SystemExit(f"{args.decisions} is empty")
    pp = per_patient(dec)
    methods = [m for m in METHOD_ORDER if m in pp]
    missing = {"mcmc", "sbi", "ukf", "linear", "neural", "prior"} - set(methods)
    if missing:
        raise SystemExit(f"the paper's claims need every arm; missing {sorted(missing)}")
    rosters = [set(pp[m]) for m in methods]
    odd = sorted(set.union(*rosters) - set.intersection(*rosters))
    if odd:
        raise SystemExit(f"ragged roster: not every arm has {odd}; cohort means and "
                         "folds would cover different patients")
    if len(rosters[0]) < MIN_PATIENTS:
        raise SystemExit(f"{len(rosters[0])} patients; the statistics need {MIN_PATIENTS}")

    full, by_patient = run(pp, methods, "patient")
    _, by_kind = run(pp, methods, "kind")
    skipped = {by: skipped_folds(pp, methods, by) for by in ("patient", "kind")}

    pairs = pair_rows(full, by_patient, methods)
    claims = (claim_rows(full, by_patient, methods)
              + [dict(r, dropped=f"kind:{r['dropped']}") for r in
                 claim_rows(full, by_kind, methods) if r["dropped"] != "<none>"])
    text = "\n".join(report(full, by_patient, by_kind, methods, args.decisions,
                            skipped)) + "\n"
    _write_csv(os.path.join(out_dir, "sensitivity_pairs.csv"), pairs)
    _write_csv(os.path.join(out_dir, "sensitivity_claims.csv"), claims)
    path = os.path.join(out_dir, "sensitivity_report.md")
    with open(path, "w") as f:
        f.write(text)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
