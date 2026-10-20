"""Critical difference diagrams for the paper — one panel per metric, stacked.

Filled dot = member of an unseparated group; open circle = crossed but excluded.
"""
from __future__ import annotations

import os
import sys
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")           # no display on the cluster or in CI
import matplotlib.pyplot as plt

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
if os.path.isdir(os.path.join(_root, "t1d_twin")) and _root not in sys.path:
    sys.path.insert(0, _root)

from experiments.analyze_benchmark import (      # noqa: E402
    METHOD_ORDER, PRETTY_METHOD, _matrix, _read, _within_patient_ranks,
    holm_pairwise, per_patient,
)

#: Display names.
PRETTY = {**PRETTY_METHOD,
          "spearman": "Spearman", "rmse": "RMSE", "mard": "MARD",
          "regret_norm": "Regret Norm", "regret": "Regret"}

#: Which end of each metric is good, for the axis annotation.
LOWER_IS_BETTER = ("regret_norm", "regret", "rmse", "mard")

# --- drawing constants -----------------------------------------------------
# Vertical geometry and fonts are in points; horizontal geometry in rank units.
FS_TICK = 7.0       # rank numbers on the axis
FS_LABEL = 7.0      # method names
FS_TAG = 8.0        # the rotated "RMSE" gutter label
LW_AXIS = 0.7       # the rank axis and its major ticks
LW_MINOR = 0.5      # half-unit ticks
LW_ELBOW = 0.5      # the leaders from the axis down to a method label
LW_BAR = 1.6        # the non-separation bars
MS_DOT = 3.2        # diameter of the member dot on a bar
MS_RING = 3.4       # diameter of the open circle where a bar crosses a non-member
LW_RING = 0.7       # its stroke
PT_TICK_MAJ = 3.5
PT_TICK_MIN = 2.0
PT_TICK_TXT = 5.0   # baseline of the rank numbers
PT_TOP = 11.0       # panel top above the axis
PT_BAR0 = -4.8      # first bar level
PT_BAR_STEP = 5.6   # between bar levels
PT_LABEL0 = -6.0    # first label row, unless a bar is in its way
PT_LABEL_STEP = 8.4 # between label rows on one side (7 pt text + leading)
PT_CLEAR = 5.0      # a label leader keeps this far from a bar it would run into
PT_BOTTOM = 4.5     # below the last label row
PT_PANEL_GAP = 4.0  # between stacked panels
X_MARGIN = 0.26     # room for the method labels, as a fraction of the rank span
X_ELBOW = 0.045     # how far past the axis ends the elbow runs, same units
X_TIE = 0.03        # display-only spread between exactly-tied mean ranks
X_BAR_GAP = 0.15    # two bars closer than this never share a level


def _display_x(R: np.ndarray, order: np.ndarray, span: float) -> np.ndarray:
    """Drawing positions: mean rank, with exactly-tied methods spread apart in ``order``."""
    x = R.astype(float).copy()
    groups: dict[float, list[int]] = {}
    for j in order:
        groups.setdefault(round(float(R[j]), 9), []).append(int(j))
    for tied in groups.values():
        if len(tied) > 1:
            for n, j in enumerate(tied):
                x[j] = R[j] + (n - (len(tied) - 1) / 2.0) * X_TIE * span
    return x


def maximal_cliques(same: np.ndarray) -> list[tuple[int, ...]]:
    """Maximal cliques of the non-separation matrix ``same`` (Bron–Kerbosch), singletons dropped."""
    k = int(same.shape[0])
    adj = [{j for j in range(k) if j != i and bool(same[i, j])} for i in range(k)]
    out: list[tuple[int, ...]] = []

    def grow(r: set, p: set, x: set) -> None:
        if not p and not x:
            if len(r) > 1:
                out.append(tuple(sorted(r)))
            return
        pivot = max(p | x, key=lambda v: len(p & adj[v]))
        for v in sorted(p - adj[pivot]):
            grow(r | {v}, p & adj[v], x & adj[v])
            p = p - {v}
            x = x | {v}

    grow(set(), set(range(k)), set())
    return sorted(out)


def bar_levels(intervals: list[tuple[float, float]]) -> list[int]:
    """First-fit level per bar; bars closer than ``X_BAR_GAP`` never share one (0 = nearest the axis)."""
    idx = sorted(range(len(intervals)), key=lambda i: intervals[i])
    levels = [0] * len(intervals)
    placed: list[list[tuple[float, float]]] = []
    for i in idx:
        lo, hi = intervals[i]
        for lvl, occupied in enumerate(placed):
            if all(hi + X_BAR_GAP <= a or b + X_BAR_GAP <= lo for a, b in occupied):
                occupied.append((lo, hi))
                levels[i] = lvl
                break
        else:
            placed.append([(lo, hi)])
            levels[i] = len(placed) - 1
    return levels


def bar_crossings(cliques: list[tuple[int, ...]], X: np.ndarray) -> list[tuple[int, ...]]:
    """Per clique, the non-members drawn strictly inside the bar's span."""
    out = []
    for c in cliques:
        lo, hi = float(X[list(c)].min()), float(X[list(c)].max())
        out.append(tuple(int(j) for j in np.argsort(X)
                         if j not in c and lo < float(X[j]) < hi))
    return out


def label_rows(side: list[int], X: np.ndarray, x_gutter: float,
               bars: list[tuple[float, float, float]]) -> dict[int, float]:
    """Label row (points) per method on one side, dropped only where its leader would hit a bar."""
    rows: dict[int, float] = {}
    y = PT_LABEL0
    for j in side:
        seg_lo, seg_hi = sorted((float(X[j]), x_gutter))
        while True:
            clash = [by for lo, hi, by in bars
                     if lo - X_BAR_GAP <= seg_hi and seg_lo <= hi + X_BAR_GAP
                     and abs(y - by) < PT_CLEAR]
            if not clash:
                break
            y = min(clash) - PT_CLEAR
        rows[j] = y
        y -= PT_LABEL_STEP
    return rows


def layout(pp: dict, methods, metric: str, alpha: float) -> dict:
    """Everything one panel needs, including its height in points, computed before drawing."""
    used, idx_pairs, _deltas, _raw, holm = holm_pairwise(pp, metric, methods)
    keep, vals, _ = _matrix(pp, methods, metric)
    if not used:
        raise SystemExit(f"no comparable data for metric {metric!r}")
    k = len(used)
    R = _within_patient_ranks(vals, metric).mean(axis=0)

    same = np.ones((k, k), dtype=bool)
    for (i, j), q in zip(idx_pairs, holm):
        same[i, j] = same[j, i] = not (np.isfinite(q) and q < alpha)

    order = np.argsort(R)                      # best (lowest mean rank) first
    span = float(k) - 1.0
    X = _display_x(R, order, span)             # draw here, label with R

    cliques = maximal_cliques(same)
    intervals = [(float(X[list(c)].min()), float(X[list(c)].max())) for c in cliques]
    levels = bar_levels(intervals)
    crossings = bar_crossings(cliques, X)
    bars = [(lo, hi, PT_BAR0 - PT_BAR_STEP * lvl)
            for (lo, hi), lvl in zip(intervals, levels)]

    # Split labels by count, not by axis midpoint; an odd extra goes left.
    half = (k + 1) // 2
    left = [int(j) for j in order[:half]]
    right = [int(j) for j in order[half:]][::-1]        # far side first
    lo_axis, hi_axis = 1.0, 1.0 + span
    rows = {**label_rows(left, X, lo_axis - X_ELBOW * span, bars),
            **label_rows(right, X, hi_axis + X_ELBOW * span, bars)}

    y_min = min([*rows.values(), *(by for _, _, by in bars)]) - PT_BOTTOM
    return {"metric": metric, "used": used, "R": R, "X": X, "order": order,
            "span": span, "same": same, "cliques": cliques, "levels": levels,
            "crossings": crossings, "bars": bars,
            "left": left, "right": right, "rows": rows,
            "y_min": y_min, "extent": PT_TOP - y_min}


def _panel(ax, lay: dict) -> None:
    """One critical difference diagram on ``ax``, from a ``layout``."""
    used, R, X, span = lay["used"], lay["R"], lay["X"], lay["span"]
    lo, hi = 1.0, 1.0 + span
    margin = X_MARGIN * span
    elbow = X_ELBOW * span
    ax.set_xlim(lo - margin, hi + margin)
    ax.set_ylim(lay["y_min"], PT_TOP)
    ax.axis("off")

    # --- rank axis -------------------------------------------------------
    ax.plot([lo, hi], [0, 0], color="black", lw=LW_AXIS, solid_capstyle="butt",
            zorder=3)
    for t in np.arange(lo, hi + 1e-9, 0.5):
        major = abs(t - round(t)) < 1e-9
        ax.plot([t, t], [0, PT_TICK_MAJ if major else PT_TICK_MIN],
                color="black", lw=LW_AXIS if major else LW_MINOR, zorder=3)
        if major:
            ax.text(t, PT_TICK_TXT, f"{t:.0f}", ha="center", va="bottom",
                    fontsize=FS_TICK)

    # --- method elbows ---------------------------------------------------
    for side, x_end, ha in ((lay["left"], lo - elbow, "right"),
                            (lay["right"], hi + elbow, "left")):
        for j in side:
            y = lay["rows"][j]
            ax.plot([X[j], X[j]], [0, y], color="black", lw=LW_ELBOW, zorder=2)
            ax.plot([X[j], x_end], [y, y], color="black", lw=LW_ELBOW, zorder=2)
            ax.text(x_end + (-0.015 if ha == "right" else 0.015) * span, y,
                    f"{PRETTY.get(used[j], used[j])} ({R[j]:.2f})",
                    ha=ha, va="center", fontsize=FS_LABEL)

    # --- non-separation bars ---------------------------------------------
    for clique, (_, _, y), crossed in zip(lay["cliques"], lay["bars"],
                                          lay["crossings"]):
        xs = sorted(float(X[j]) for j in clique)
        ax.plot(xs, [y] * len(xs), color="black", lw=LW_BAR, marker="o",
                markersize=MS_DOT, markeredgewidth=0, solid_capstyle="round",
                zorder=4)
        for j in crossed:
            ax.plot([X[j]], [y], linestyle="none", marker="o",
                    markersize=MS_RING, markerfacecolor="white",
                    markeredgecolor="black", markeredgewidth=LW_RING,
                    zorder=5)

    # --- panel tag, rotated in the left gutter outside the axes ----------
    ax.text(-0.012, 0.5, PRETTY.get(lay["metric"], lay["metric"]),
            transform=ax.transAxes, rotation=90, rotation_mode="anchor",
            ha="center", va="bottom", fontsize=FS_TAG)


def main() -> None:
    """CLI: draw the critical difference diagram as a paper figure."""
    from experiments import output_paths as OP
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--decisions", default=None,
                    help="benchmark_decisions.csv (default: the dated path)")
    ap.add_argument("--out", default=None,
                    help="output figure; .pdf is what the paper wants")
    ap.add_argument("--metrics", default="rmse,spearman",
                    help="comma-separated, one panel each, top to bottom")
    ap.add_argument("--methods", default=None,
                    help="comma-separated subset to draw, in CSV keys "
                         "(default: every method in the table). The Holm "
                         "correction is computed over exactly these, so this "
                         "must match the roster the paper reports")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--width", type=float, default=5.5,
                    help="inches; 5.5 is the paper's textwidth, so the figure "
                         "is 1:1 under width=\\linewidth. Height follows content")
    args = ap.parse_args()

    dec_path = args.decisions or OP.decisions_table_path()
    out_path = args.out or os.path.join(
        os.path.dirname(OP.benchmark_report_path()), "critical_difference_diagram.pdf")
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    dec = _read(dec_path)
    if not dec:
        raise SystemExit(f"{dec_path} is empty")
    pp = per_patient(dec)
    methods = ([m for m in METHOD_ORDER if m in pp]
               + [m for m in pp if m not in METHOD_ORDER])
    if args.methods:
        want = [m.strip() for m in args.methods.split(",") if m.strip()]
        missing = [m for m in want if m not in pp]
        if missing:
            raise SystemExit(f"--methods not in {dec_path}: {', '.join(missing)}")
        methods = [m for m in methods if m in want]

    lays = [layout(pp, methods, metric, args.alpha) for metric in metrics]
    extents = [lay["extent"] for lay in lays]

    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "Nimbus Roman",
                                        "Liberation Serif", "STIXGeneral",
                                        "DejaVu Serif"],
                         "mathtext.fontset": "stix",
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    # Each axes is exactly its extent tall, so one y unit is one point.
    n = len(lays)
    fig_h = (sum(extents) + PT_PANEL_GAP * (n - 1)) / 72.0
    fig, axes = plt.subplots(n, 1, figsize=(args.width, fig_h),
                             gridspec_kw={"height_ratios": extents})
    axes = np.atleast_1d(axes)
    for ax, lay in zip(axes, lays):
        _panel(ax, lay)
    # The left inset is the gutter the rotated panel tags live in.
    fig.subplots_adjust(left=0.035, right=0.998, top=1.0, bottom=0.0,
                        hspace=PT_PANEL_GAP / (sum(extents) / n))

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path)
    png = os.path.splitext(out_path)[0] + ".png"
    fig.savefig(png, dpi=300)
    plt.close(fig)

    for lay in lays:
        name = lambda j: PRETTY.get(lay["used"][j], lay["used"][j])  # noqa: E731
        groups = ", ".join(
            "{" + ", ".join(name(j) for j in c) + "}"
            + (" over " + "/".join(name(j) for j in x) if x else "")
            for c, x in zip(lay["cliques"], lay["crossings"])) or "none"
        print(f"[plot] {lay['metric']}: {len(lay['cliques'])} bars on "
              f"{(max(lay['levels']) + 1) if lay['levels'] else 0} levels, "
              f"{lay['extent']:.1f} pt tall: {groups}")
    print(f"[plot] {n} panels ({', '.join(metrics)}) over {len(methods)} "
          f"methods from {dec_path}; {args.width:.2f} x {fig_h:.2f} in")
    print(f"[plot] -> {out_path}")
    print(f"[plot] -> {png}")
    print(f"[plot] LaTeX: \\includegraphics[width=\\linewidth]"
          f"{{{os.path.splitext(os.path.basename(out_path))[0]}}}")


if __name__ == "__main__":
    main()
