"""Merge per-arm benchmark tables from several experiment directories into one."""
from __future__ import annotations

import argparse
import csv
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
if os.path.isdir(os.path.join(_root, "t1d_twin")) and _root not in sys.path:
    sys.path.insert(0, _root)

from experiments import output_paths as OP        # noqa: E402

LONG_NAME = "benchmark_long.csv"
DEC_NAME = "benchmark_decisions.csv"
REPORT_NAME = "benchmark_report.md"


def results_root() -> str:
    """``results/<date>/`` — the parent every experiment namespace sits under."""
    return os.path.dirname(OP.RESULTS_DIR.rstrip(os.sep))


def resolve_source(spec: str) -> str:
    """A source spec -> a directory holding the two CSVs."""
    if os.path.isabs(spec):
        return spec
    if "/" in spec:
        date, _, ns = spec.rpartition("/")
        root = os.path.dirname(results_root().rstrip(os.sep))   # -> results/
        return os.path.join(root, date, ns)
    return os.path.join(results_root(), spec)


def source_label(spec: str) -> str:
    """Short name for a source, for the duplicate-arm and drift messages."""
    return spec.rstrip("/").split("/")[-1] or spec


def _read(path: str) -> tuple[list[dict], list[str]]:
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        return [dict(row) for row in r], list(r.fieldnames or [])


def _write(path: str, rows: list[dict], columns: list[str]) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in columns})
    return path


def merge_tables(tables: list[tuple[str, list[dict], list[str]]],
                 key_extra: tuple[str, ...] = ()) -> tuple[list[dict], list[str]]:
    """Union rows from ``[(source_name, rows, columns), ...]`` with validation."""
    if not tables:
        raise ValueError("nothing to merge")

    base_cols = tables[0][2]
    for name, _, cols in tables[1:]:
        if set(cols) != set(base_cols):
            missing = sorted(set(base_cols) - set(cols))
            extra = sorted(set(cols) - set(base_cols))
            raise ValueError(
                f"schema drift in {name!r}: missing {missing}, unexpected {extra}. "
                f"These tables were written by different code — re-run the odd one "
                f"out rather than merging across a schema change.")

    seen_methods: dict[str, str] = {}
    for name, rows, _ in tables:
        for m in {r["method"] for r in rows}:
            if m in seen_methods:
                raise ValueError(
                    f"method {m!r} appears in BOTH {seen_methods[m]!r} and "
                    f"{name!r}. Merging it twice would pair the method against "
                    f"itself; drop one source.")
            seen_methods[m] = name

    merged: list[dict] = []
    for _, rows, _ in tables:
        merged.extend(rows)

    key = ("patient", "day", "method") + key_extra
    keys = [tuple(r.get(k, "") for k in key) for r in merged]
    if len(set(keys)) != len(keys):
        dupes = {k for k in keys if keys.count(k) > 1}
        raise ValueError(f"{len(dupes)} duplicate rows on {key}: "
                         f"{sorted(dupes)[:3]}")
    return merged, base_cols


def check_grid(rows: list[dict]) -> dict[str, int]:
    """Every method must cover the same (patient, day) grid. Returns per-method n."""
    grids: dict[str, set] = {}
    for r in rows:
        grids.setdefault(r["method"], set()).add((r["patient"], r["day"]))
    if not grids:
        raise ValueError("no rows to check")

    ref_m, ref = next(iter(grids.items()))
    for m, g in grids.items():
        if g != ref:
            only_ref = sorted({p for p, _ in ref - g})[:5]
            only_m = sorted({p for p, _ in g - ref})[:5]
            raise ValueError(
                f"method {m!r} covers a different (patient, day) grid than "
                f"{ref_m!r}: {len(g)} vs {len(ref)} cells. "
                f"missing here: {only_ref}; extra here: {only_m}. "
                f"The paired analysis would compare different cohorts.")
    return {m: len(g) for m, g in grids.items()}


def merge(sources: list[str], into: str, analyze: bool = False,
          allow_smoke: bool = False) -> dict:
    """Merge ``sources`` (experiment dir names) into ``into``. Returns a summary."""
    # A smoke namespace holds numbers from reduced budgets.
    if not allow_smoke:
        smoky = [s for s in sources
                 if source_label(s) == "smoke" or source_label(s).endswith("_smoke")]
        if smoky:
            raise SystemExit(
                f"refusing to merge smoke namespace(s) {smoky}.\n"
                f"  Smoke runs use reduced budgets and are not results. Re-run "
                f"the arm without --smoke, or pass allow_smoke=True if you are "
                f"deliberately assembling a throwaway.")
    root = results_root()
    long_in, dec_in = [], []
    for s in sources:
        d = resolve_source(s)
        lp, dp = os.path.join(d, LONG_NAME), os.path.join(d, DEC_NAME)
        for p in (lp, dp):
            if not os.path.exists(p):
                raise SystemExit(
                    f"missing {p}\n"
                    f"  source {s!r} has no results there. If this arm was run "
                    f"under a DIFFERENT date, name it explicitly as "
                    f"'<date>/<namespace>' rather than relying on --date.")
        rows, cols = _read(lp)
        long_in.append((source_label(s), rows, cols))
        rows, cols = _read(dp)
        dec_in.append((source_label(s), rows, cols))

    long_rows, long_cols = merge_tables(long_in, key_extra=("therapy",))
    dec_rows, dec_cols = merge_tables(dec_in)
    per_method = check_grid(dec_rows)

    out_dir = os.path.join(root, into)
    lp = _write(os.path.join(out_dir, LONG_NAME), long_rows, long_cols)
    dp = _write(os.path.join(out_dir, DEC_NAME), dec_rows, dec_cols)

    summary = {
        "sources": sources, "out_dir": out_dir,
        "long_rows": len(long_rows), "decision_rows": len(dec_rows),
        "methods": sorted(per_method), "cells_per_method": per_method,
        "patients": len({r["patient"] for r in dec_rows}),
        "days": len({r["day"] for r in dec_rows}),
        "long_path": lp, "decisions_path": dp, "report_path": None,
    }

    if analyze:
        from experiments import analyze_benchmark as AB
        report = os.path.join(out_dir, REPORT_NAME)
        argv = sys.argv
        try:
            sys.argv = ["analyze_benchmark", "--decisions", dp, "--long", lp,
                        "--out", report]
            AB.main()
        finally:
            sys.argv = argv
        summary["report_path"] = report
    return summary


def main() -> None:
    """CLI: union per-arm result namespaces into one combined vintage."""
    ap = argparse.ArgumentParser(
        description="Union per-arm benchmark tables into one combined vintage.")
    ap.add_argument("--sources", default="benchmark,prior,neural",
                    help="comma-separated sources. Each is a namespace under "
                         "results/<date>/, or '<date>/<namespace>' to name a "
                         "different vintage, or an absolute path.")
    ap.add_argument("--into", default="combined",
                    help="destination experiment dir name (created)")
    ap.add_argument("--analyze", action="store_true",
                    help="also run analyze_benchmark over the merged table")
    ap.add_argument("--allow-smoke", action="store_true",
                    help="permit *_smoke sources (reduced budgets; NOT results)")
    args = ap.parse_args()

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    if any(resolve_source(s) == os.path.join(results_root(), args.into)
           for s in sources):
        raise SystemExit(f"--into {args.into!r} is also a source; that would "
                         f"overwrite an input")

    s = merge(sources, args.into, analyze=args.analyze,
              allow_smoke=args.allow_smoke)
    print(f"[merge] sources: {', '.join(s['sources'])}")
    print(f"[merge] methods: {', '.join(s['methods'])}  "
          f"({s['patients']} patients x {s['days']} days)")
    for m, n in sorted(s["cells_per_method"].items()):
        print(f"           {m:8s} {n} patient-days")
    print(f"[merge] {s['long_rows']} long rows   -> {s['long_path']}")
    print(f"[merge] {s['decision_rows']} decision rows -> {s['decisions_path']}")
    if s["report_path"]:
        print(f"[merge] report -> {s['report_path']}")
    else:
        print(f"[merge] now run:  python -m experiments.analyze_benchmark "
              f"--decisions {s['decisions_path']} --long {s['long_path']} "
              f"--out {os.path.join(s['out_dir'], REPORT_NAME)}")


if __name__ == "__main__":
    main()
