"""Significance diagrams for the paper — one panel per metric, stacked."""
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
# All in axis-y units except the fonts (points) and the figure size (inches).
FS_TICK = 7.0       # rank numbers on the axis
FS_LABEL = 7.0      # method names
FS_TAG = 8.0        # the rotated "(a) Spearman" gutter label
LW_AXIS = 0.7       # the rank axis and its major ticks
LW_MINOR = 0.5      # half-unit ticks
LW_ELBOW = 0.5      # the leaders from the axis down to a method label
LW_BAR = 1.7        # the non-significance bars
Y_TICK_MAJ = 0.045
Y_TICK_MIN = 0.028
Y_TICK_TXT = 0.065
Y_BAR0 = -0.085     # first bar, just under the axis
Y_BAR_STEP = 0.062
Y_ELBOW0 = -0.20    # first method elbow
Y_ELBOW_STEP = 0.115
X_MARGIN = 0.26     # room for the method labels, as a fraction of the rank span
X_ELBOW = 0.045     # how far past the axis ends the elbow runs, same units


def _panel(ax, pp: dict, methods, metric: str, alpha: float) -> None:
    """One significance diagram on ``ax``."""
    used, idx_pairs, _deltas, _raw, holm = holm_pairwise(pp, metric, methods)
    keep, vals, _ = _matrix(pp, methods, metric)
    if not used:
        raise SystemExit(f"no comparable data for metric {metric!r}")
    k, N = len(used), len(keep)
    R = _within_patient_ranks(vals, metric).mean(axis=0)

    same = np.ones((k, k), dtype=bool)
    for (i, j), q in zip(idx_pairs, holm):
        same[i, j] = same[j, i] = not (np.isfinite(q) and q < alpha)

    order = np.argsort(R)                      # best (lowest mean rank) first
    lo, hi = 1.0, float(k)
    span = hi - lo
    margin = X_MARGIN * span
    elbow = X_ELBOW * span
    ax.set_xlim(lo - margin, hi + margin)
    ax.axis("off")
    y_min = -0.24

    # --- rank axis -------------------------------------------------------
    ax.plot([lo, hi], [0, 0], color="black", lw=LW_AXIS, solid_capstyle="butt",
            zorder=3)
    for t in np.arange(lo, hi + 1e-9, 0.5):
        major = abs(t - round(t)) < 1e-9
        ax.plot([t, t], [0, Y_TICK_MAJ if major else Y_TICK_MIN], color="black",
                lw=LW_AXIS if major else LW_MINOR, zorder=3)
        if major:
            ax.text(t, Y_TICK_TXT, f"{t:.0f}", ha="center", va="bottom",
                    fontsize=FS_TICK)

    # --- method elbows ---------------------------------------------------
    # Split by COUNT, not by axis midpoint: mean ranks cluster toward the good
    # end, so a midpoint split leaves one side crowded and the other nearly
    # empty. Ceil goes left, so an odd roster puts the extra method with the
    # better-ranked group.
    half = (len(order) + 1) // 2
    left = list(order[:half])
    right = list(order[half:])[::-1]                    # far side first
    depth = {}
    for side in (left, right):
        for n, j in enumerate(side):
            depth[j] = Y_ELBOW0 - Y_ELBOW_STEP * n
            y_min = min(y_min, depth[j])
    for side, x_end, ha in ((left, lo - elbow, "right"),
                            (right, hi + elbow, "left")):
        for j in side:
            y = depth[j]
            ax.plot([R[j], R[j]], [0, y], color="black", lw=LW_ELBOW, zorder=2)
            ax.plot([R[j], x_end], [y, y], color="black", lw=LW_ELBOW, zorder=2)
            ax.text(x_end + (-0.015 if ha == "right" else 0.015) * span, y,
                    f"{PRETTY.get(used[j], used[j])} ({R[j]:.2f})",
                    ha=ha, va="center", fontsize=FS_LABEL)

    # --- non-significance bars ------------------------------------------
    # Maximal runs, contiguous in mean-rank order, in which EVERY pair is non-
    # significant.
    drawn, y_bar = set(), Y_BAR0
    for i in range(k):
        j = i
        while j + 1 < k and all(same[order[a], order[b]]
                                for a in range(i, j + 2)
                                for b in range(a + 1, j + 2)):
            j += 1
        if j > i and (order[i], order[j]) not in drawn:
            drawn.add((order[i], order[j]))
            ax.plot([R[order[i]], R[order[j]]], [y_bar, y_bar], color="black",
                    lw=LW_BAR, solid_capstyle="round", zorder=4)
            y_min = min(y_min, y_bar)
            y_bar -= Y_BAR_STEP

    # --- panel tag -------------------------------------------------------
    # Runs vertically in the left gutter, OUTSIDE the axes, so it costs the drawing no
    # width and no height.
    ax.text(-0.012, 0.5, PRETTY.get(metric, metric),
            transform=ax.transAxes, rotation=90, rotation_mode="anchor",
            ha="center", va="bottom", fontsize=FS_TAG)

    # Now that everything is drawn, size the panel to what it actually contains.
    ax.set_ylim(y_min - 0.07, 0.155)


def main() -> None:
    """CLI: draw the significance diagram as a paper figure."""
    from experiments import output_paths as OP
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--decisions", default=None,
                    help="benchmark_decisions.csv (default: the dated path)")
    ap.add_argument("--out", default=None,
                    help="output figure; .pdf is what the paper wants")
    ap.add_argument("--metrics", default="spearman,rmse",
                    help="comma-separated, one panel each, top to bottom")
    ap.add_argument("--methods", default=None,
                    help="comma-separated subset to draw, in CSV keys "
                         "(default: every method in the table). The Holm "
                         "correction is computed over exactly these, so this "
                         "must match the roster the paper reports")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--width", type=float, default=5.5,
                    help="inches; 5.5 is the neurips_2026 textwidth, so a "
                         "width=\\linewidth include is 1:1 and the point sizes "
                         "here are the point sizes on the page")
    ap.add_argument("--panel-height", type=float, default=None,
                    help="inches; default scales with the number of methods "
                         "(0.62 at k<=3, +0.13 per extra arm) so the deepest "
                         "elbow label is never clipped")
    args = ap.parse_args()

    dec_path = args.decisions or OP.decisions_table_path()
    out_path = args.out or os.path.join(
        os.path.dirname(OP.benchmark_report_path()), "significance_diagram.pdf")
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

    # Panel height scales with the roster unless the caller pins it.
    panel_h = (args.panel_height if args.panel_height is not None
               else 0.62 + 0.13 * max(0, len(methods) - 3))

    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "Nimbus Roman",
                                        "Liberation Serif", "STIXGeneral",
                                        "DejaVu Serif"],
                         "mathtext.fontset": "stix",
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(len(metrics), 1,
                             figsize=(args.width, panel_h * len(metrics)))
    axes = np.atleast_1d(axes)
    for ax, metric in zip(axes, metrics):
        _panel(ax, pp, methods, metric, args.alpha)
    # The left inset is the gutter the rotated panel tags live in; everything else is
    # trimmed to the drawing.
    fig.subplots_adjust(left=0.035, right=0.998, top=0.99, bottom=0.01,
                        hspace=0.10)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path)
    png = os.path.splitext(out_path)[0] + ".png"
    fig.savefig(png, dpi=300)
    plt.close(fig)

    print(f"[plot] {len(metrics)} panels ({', '.join(metrics)}) "
          f"over {len(methods)} methods from {dec_path}")
    print(f"[plot] -> {out_path}")
    print(f"[plot] -> {png}")
    print(f"[plot] LaTeX: \\includegraphics[width=\\linewidth]"
          f"{{{os.path.splitext(os.path.basename(out_path))[0]}}}")


if __name__ == "__main__":
    main()
