"""Analysis of the benchmark long tables -> ``results/<date>/benchmark_report.md``."""
from __future__ import annotations

import os
import sys
import csv
import argparse
import itertools
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon, spearmanr, friedmanchisquare

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
if os.path.isdir(os.path.join(_root, "t1d_twin")) and _root not in sys.path:
    sys.path.insert(0, _root)

#: Metrics where a LOWER value is better.
LOWER_IS_BETTER = ("regret_norm", "regret", "rmse", "mard")

METRICS = ("spearman", "regret_norm", "regret", "rmse", "mard")

#: Metrics the omnibus comparison runs on.
OMNIBUS_METRICS = ("spearman", "regret_norm", "rmse", "mard")

#: Canonical DISPLAY order for the arms, used by both the report and the paper figure
#: (``plot_critical_difference``).
METHOD_ORDER = ("mcmc", "sbi", "ukf", "linear", "neural", "prior")

#: Display names for the arms.
PRETTY_METHOD = {"mcmc": "MCMC", "sbi": "SBI", "ukf": "UKF",
                 "linear": "Linear", "neural": "Neural", "prior": "Prior"}


def _read(path):
    with open(path) as f:
        return [dict(r) for r in csv.DictReader(f)]


def _f(row, key):
    v = row.get(key, "")
    if v in ("", None):
        return np.nan
    try:
        return float(v)
    except ValueError:
        return np.nan


def per_patient(dec_rows) -> dict:
    """``method -> patient -> {metric: mean over that patient's test days}``."""
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in dec_rows:
        m, p = r["method"], r["patient"]
        for k in METRICS:
            acc[m][p][k].append(_f(r, k))
        acc[m][p]["degenerate"].append(_f(r, "degenerate"))
        nf = _f(r, "nonfinite")
        acc[m][p]["nonfinite"].append(0.0 if not np.isfinite(nf) else nf)
    out = {}
    for m, pats in acc.items():
        out[m] = {p: {k: float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
                      for k, v in d.items()}
                  for p, d in pats.items()}
    return out


def _fmt(x, nd=3):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def method_table(pp: dict, methods) -> list[str]:
    """Markdown table of cohort-mean metrics, one row per method."""
    lines = ["| method | n | spearman | regret_norm | regret | RMSE | MARD | degenerate | nonfinite |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for m in methods:
        d = pp.get(m, {})
        if not d:
            continue
        col = {k: np.array([v[k] for v in d.values()], float) for k in METRICS}
        deg = np.array([v.get("degenerate", 0.0) for v in d.values()], float)
        nfn = np.array([v.get("nonfinite", 0.0) for v in d.values()], float)
        lines.append(
            f"| {m} | {len(d)} | "
            f"{_fmt(np.nanmean(col['spearman']))} ± {_fmt(np.nanstd(col['spearman']), 2)} | "
            f"{_fmt(np.nanmean(col['regret_norm']))} | "
            f"{_fmt(np.nanmean(col['regret']), 1)} | "
            f"{_fmt(np.nanmean(col['rmse']), 1)} | "
            f"{_fmt(np.nanmean(col['mard']), 1)} | "
            f"{_fmt(100 * np.nanmean(deg), 1)}% | "
            f"{_fmt(np.nansum(nfn), 0)} |")
    return lines


def paired_table(pp: dict, methods) -> list[str]:
    """Markdown table of per-pair mean differences and Wilcoxon p-values."""
    lines = ["| A vs B | n paired | Δ spearman (A−B) | Δ regret_norm | Wilcoxon p (spearman) |",
             "|---|---:|---:|---:|---:|"]
    for i, a in enumerate(methods):
        for b in methods[i + 1:]:
            da, db = pp.get(a, {}), pp.get(b, {})
            shared = sorted(set(da) & set(db))
            if len(shared) < 3:
                continue
            sa = np.array([da[p]["spearman"] for p in shared], float)
            sb = np.array([db[p]["spearman"] for p in shared], float)
            ra = np.array([da[p]["regret_norm"] for p in shared], float)
            rb = np.array([db[p]["regret_norm"] for p in shared], float)
            ok = np.isfinite(sa) & np.isfinite(sb)
            p_val = np.nan
            if ok.sum() >= 3 and np.ptp((sa - sb)[ok]) > 0:
                try:
                    p_val = float(wilcoxon(sa[ok], sb[ok]).pvalue)
                except ValueError:
                    p_val = np.nan
            lines.append(f"| {a} vs {b} | {int(ok.sum())} | "
                         f"{_fmt(np.nanmean(sa - sb))} | "
                         f"{_fmt(np.nanmean(ra - rb))} | {_fmt(p_val, 4)} |")
    return lines


# ---------------------------------------------------------------------------
# Omnibus comparison across k methods (Demsar 2006)
# ---------------------------------------------------------------------------
def _matrix(pp: dict, methods, metric: str):
    """``(patients, values (P, k), methods_used)`` on the patients ALL methods share."""
    used = [m for m in methods if pp.get(m)]
    if len(used) < 2:
        return [], np.zeros((0, 0)), used
    shared = sorted(set.intersection(*[set(pp[m]) for m in used]))
    rows, keep = [], []
    for p in shared:
        v = [pp[m][p].get(metric, np.nan) for m in used]
        if all(np.isfinite(x) for x in v):
            rows.append(v)
            keep.append(p)
    return keep, np.asarray(rows, float), used


def _friedman(vals: np.ndarray) -> float:
    """Friedman p, or NaN when the statistic is undefined."""
    if vals.ndim != 2 or vals.shape[0] < 3 or vals.shape[1] < 3:
        return float("nan")
    if np.allclose(vals.max(axis=1) - vals.min(axis=1), 0.0):
        return float("nan")
    try:
        with np.errstate(invalid="ignore", divide="ignore"):
            return float(friedmanchisquare(
                *[vals[:, j] for j in range(vals.shape[1])]).pvalue)
    except ValueError:
        return float("nan")


def _within_patient_ranks(vals: np.ndarray, metric: str) -> np.ndarray:
    """Rank methods within each patient; rank 1 = best. Ties get the average."""
    from scipy.stats import rankdata
    x = vals if metric in LOWER_IS_BETTER else -vals
    return np.vstack([rankdata(row, method="average") for row in x])


def omnibus_table(pp: dict, methods) -> list[str]:
    """Friedman per metric, with mean within-patient ranks."""
    lines = ["| metric | n patients | k methods | " +
             " | ".join(f"mean rank: {m}" for m in methods) + " | Friedman p |",
             "|---" * (3 + len(methods) + 1) + "|"]
    any_row = False
    for metric in OMNIBUS_METRICS:
        keep, vals, used = _matrix(pp, methods, metric)
        if len(keep) < 3 or len(used) < 3:
            continue
        R = _within_patient_ranks(vals, metric)
        p = _friedman(vals)
        cells = {m: "" for m in methods}
        for j, m in enumerate(used):
            cells[m] = _fmt(R[:, j].mean(), 2)
        lines.append(f"| {metric} | {len(keep)} | {len(used)} | " +
                     " | ".join(cells[m] or "—" for m in methods) +
                     f" | {_fmt(p, 4)} |")
        any_row = True
    return lines if any_row else []


def holm_pairwise(pp: dict, metric: str, methods):
    """Pairwise Wilcoxon signed-rank with a Holm correction."""
    keep, vals, used = _matrix(pp, methods, metric)
    if len(keep) < 3 or len(used) < 2:
        return [], [], [], [], []
    idx_pairs, deltas, raw = [], [], []
    for i, j in itertools.combinations(range(len(used)), 2):
        a, b = vals[:, i], vals[:, j]
        try:
            p = float(wilcoxon(a, b).pvalue) if np.ptp(a - b) > 0 else 1.0
        except ValueError:
            p = np.nan
        idx_pairs.append((i, j))
        deltas.append(float(np.mean(a - b)))
        raw.append(p)
    order = np.argsort(np.where(np.isfinite(raw), raw, np.inf))
    holm = np.full(len(raw), np.nan)
    running = 0.0
    for rank, idx in enumerate(order):
        if not np.isfinite(raw[idx]):
            continue
        running = max(running, min(1.0, raw[idx] * (len(raw) - rank)))
        holm[idx] = running
    return used, idx_pairs, deltas, raw, holm


def posthoc_table(pp: dict, methods, metric: str, alpha: float = 0.05) -> list[str]:
    """Pairwise Wilcoxon with Holm correction. Run only if Friedman rejects."""
    used, idx_pairs, deltas, raw, holm = holm_pairwise(pp, metric, methods)
    if not used:
        return []
    lines = [f"| A vs B | Δ {metric} (A−B) | raw p | Holm p | sig at {alpha} |",
             "|---|---:|---:|---:|:--:|"]
    for (i, j), d, p, q in zip(idx_pairs, deltas, raw, holm):
        mark = "yes" if np.isfinite(q) and q < alpha else "no"
        lines.append(f"| {used[i]} vs {used[j]} | {_fmt(d)} | {_fmt(p, 4)} | "
                     f"{_fmt(q, 4)} | {mark} |")
    return lines


def critical_difference_diagram(pp: dict, methods, metric: str,
                                alpha: float = 0.05) -> list[str]:
    """ASCII critical difference diagram: mean ranks, unseparated methods joined."""
    used, idx_pairs, deltas, raw, holm = holm_pairwise(pp, metric, methods)
    if not used:
        return []
    keep, vals, _ = _matrix(pp, methods, metric)
    k, N = len(used), len(keep)
    R = _within_patient_ranks(vals, metric).mean(axis=0)

    # Symmetric "not significantly different" lookup.
    same = np.ones((k, k), dtype=bool)
    for (i, j), q in zip(idx_pairs, holm):
        sig = bool(np.isfinite(q) and q < alpha)
        same[i, j] = same[j, i] = not sig

    order = np.argsort(R)                       # best (lowest mean rank) first
    lines = ["```",
             f"critical difference diagram — {metric}   (N={N} patients, k={k} methods)",
             f"bars = NOT significantly different, pairwise Wilcoxon + Holm at "
             f"alpha={alpha}",
             "rank 1 = best; the axis position is the mean within-patient rank.",
             ""]
    lo, hi = 1.0, float(k)
    width = 56

    def pos(r):
        """Map a mean rank onto a column of the text axis."""
        return int(round((r - lo) / max(hi - lo, 1e-9) * (width - 1)))

    axis = [" "] * width
    for tick in range(1, k + 1):
        t = pos(tick)
        if 0 <= t < width:
            axis[t] = "|"
    lines.append("  " + "".join(axis))
    lines.append("  " + "-" * width)
    labels = [" "] * width
    for tick in range(1, k + 1):
        t = pos(tick)
        for c, ch in enumerate(str(tick)):
            if 0 <= t + c < width:
                labels[t + c] = ch
    lines.append("  " + "".join(labels))
    lines.append("")
    for j in order:
        lines.append("  " + " " * pos(R[j]) + f"^ {used[j]}  (mean rank {R[j]:.2f})")

    # Bars: maximal runs, contiguous in mean-rank order, in which EVERY pair is non-
    # significant.
    lines.append("")
    drawn, bars = set(), 0
    for i in range(k):
        j = i
        while j + 1 < k and all(same[order[a], order[b]]
                                for a in range(i, j + 2)
                                for b in range(a + 1, j + 2)):
            j += 1
        if j > i:
            key = (order[i], order[j])
            if key in drawn:
                continue
            drawn.add(key)
            a, b = pos(R[order[i]]), pos(R[order[j]])
            bar = [" "] * width
            for c in range(min(a, b), max(a, b) + 1):
                bar[c] = "="
            names = ", ".join(used[order[t]] for t in range(i, j + 1))
            lines.append("  " + "".join(bar) + f"  not distinguishable: {names}")
            bars += 1
    if not bars:
        lines.append("  (every pair differs significantly)")
    lines.append("```")
    return lines

def dt2_table(pp: dict, methods) -> list[str]:
    """Across patients, does fidelity predict decision quality?"""
    lines = ["| method | n | rho(−MARD, spearman) | p | rho(−RMSE, spearman) | p |",
             "|---|---:|---:|---:|---:|---:|"]
    for m in methods:
        d = pp.get(m, {})
        sp = np.array([v["spearman"] for v in d.values()], float)
        md = np.array([v["mard"] for v in d.values()], float)
        rm = np.array([v["rmse"] for v in d.values()], float)
        ok = np.isfinite(sp) & np.isfinite(md) & np.isfinite(rm)
        if ok.sum() < 4:
            continue
        r1 = spearmanr(-md[ok], sp[ok])
        r2 = spearmanr(-rm[ok], sp[ok])
        lines.append(f"| {m} | {int(ok.sum())} | {_fmt(r1.correlation)} | "
                     f"{_fmt(r1.pvalue, 4)} | {_fmt(r2.correlation)} | "
                     f"{_fmt(r2.pvalue, 4)} |")
    return lines


def winner_table(long_rows) -> list[str]:
    """Which therapy actually wins, per patient-day."""
    # Group once.
    by_cell = defaultdict(list)
    for r in long_rows:
        by_cell[(r["patient"], r["day"])].append(r)
    best = defaultdict(int)
    for rows in by_cell.values():
        # true_reward is a property of (patient, day, therapy) and is identical across
        # methods, so dedupe on therapy before taking the argmax.
        per_therapy = {x["therapy"]: _f(x, "true_reward") for x in rows}
        if per_therapy:
            best[max(per_therapy, key=per_therapy.get)] += 1
    tot = sum(best.values()) or 1
    lines = ["| winning therapy | patient-days | share |", "|---|---:|---:|"]
    for k, v in sorted(best.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {k} | {v} | {100 * v / tot:.1f}% |")
    return lines


def main() -> None:
    """CLI: read the shipped tables and write the benchmark report."""
    from experiments import output_paths as OP
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--decisions", default=None)
    ap.add_argument("--long", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dec_path = args.decisions or OP.decisions_table_path()
    long_path = args.long or OP.long_table_path()
    out_path = args.out or OP.benchmark_report_path()

    dec = _read(dec_path)
    if not dec:
        raise SystemExit(f"{dec_path} is empty")
    long_rows = _read(long_path) if os.path.exists(long_path) else []

    pp = per_patient(dec)
    methods = ([m for m in METHOD_ORDER if m in pp]
               + [m for m in pp if m not in METHOD_ORDER])
    n_pat = len({r["patient"] for r in dec})
    n_day = len({r["day"] for r in dec})

    L = [f"# Benchmark results", "",
         f"source: `{dec_path}`", "",
         f"- patients: **{n_pat}**   test days: **{n_day}**   "
         f"methods: {', '.join(methods)}",
         f"- unit of analysis: the **patient** (days averaged within patient "
         f"first); n = {n_pat}, not {n_pat * n_day}", "",
         "## 1. Per-method performance", "",
         "spearman is the primary metric. `degenerate` is the share of "
         "patient-days on which a twin predicted identical rewards for every "
         "therapy — spearman is undefined there and those days are excluded "
         "from its mean, so a high value invalidates the row above it. "
         "`nonfinite` is the total count of (patient, day, therapy) cells whose "
         "predicted reward was NaN or infinite; **it must be 0**. Anything else "
         "means that method failed to produce a number on some cells, and every "
         "metric on its row is then computed over a subset of the grid.", ""]
    L += method_table(pp, methods)
    L += ["", "## 2. Method comparisons", "",
          "Paired on the patient. Patient difficulty dominates the between-method "
          "effect, so unpaired tests here would mostly measure cohort composition.",
          ""]
    if len(methods) < 3:
        L += ["With two methods the omnibus test degenerates to a sign test, so "
              "this is a paired Wilcoxon signed-rank (two-sided).", ""]
        L += paired_table(pp, methods)
    else:
        L += ["With three or more methods, an all-pairs comparison would cost "
              f"{len(methods) * (len(methods) - 1) // 2} tests and lose power to "
              "the multiplicity correction. Instead: an omnibus **Friedman** test "
              "per metric on the within-patient ranks (rank 1 = best), then "
              "post-hoc pairwise Wilcoxon with Holm correction, and only where "
              "Friedman rejects. Protocol from Demsar 2006.", ""]
        L += omnibus_table(pp, methods)
        for metric in OMNIBUS_METRICS:
            keep, vals, used = _matrix(pp, methods, metric)
            if len(keep) < 3 or len(used) < 3:
                continue
            p = _friedman(vals)
            if not (np.isfinite(p) and p < 0.05):
                continue
            L += ["", f"### 2.{OMNIBUS_METRICS.index(metric) + 1} post-hoc — "
                      f"`{metric}` (Friedman p = {_fmt(p, 4)})", ""]
            L += posthoc_table(pp, methods, metric)
        L += ["", "### Critical difference diagram", "",
              "Demšar's critical difference diagram in the Wilcoxon–Holm form: "
              "mean within-patient rank, best on the left. Methods joined by a "
              "bar are NOT significantly different by pairwise Wilcoxon with a "
              "Holm correction — the same test as the post-hoc tables above, so "
              "the diagram and the tables cannot disagree. No Nemenyi "
              "critical-difference interval is drawn; rank only positions a "
              "method on the axis and decides nothing.",
              ""]
        for metric in ("spearman", "rmse"):
            L += critical_difference_diagram(pp, methods, metric)
            L += [""]
    L += ["", "## 3. DT² — does trajectory fidelity predict decision quality?", "",
          "Rank correlation across patients between fidelity (negated, so higher "
          "is better) and decision quality. **A correlation near zero is the "
          "project's central claim**: a twin can track glucose closely and still "
          "rank therapies wrongly. A strongly positive correlation would mean "
          "fidelity is a sufficient proxy and the benchmark has nothing to add.",
          ""]
    L += dt2_table(pp, methods)
    if long_rows:
        L += ["", "## 4. Which therapy actually wins", "",
              "The design originally assumed a universal optimum at `bolus_r1.00`, "
              "which would make regret degenerate. It is not universal — the Magni "
              "kernel penalises hypoglycemia far more steeply than hyperglycemia, "
              "so the risk-optimal dose sits below the carb-ratio-correct one.", ""]
        L += winner_table(long_rows)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n[analyze] -> {out_path}")


if __name__ == "__main__":
    main()
