"""Single source of truth for the standardized output directory layout."""
from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Root + top-level trees
# ---------------------------------------------------------------------------
OUTPUT_ROOT = os.environ.get("T1D_OUTPUT_ROOT", os.path.join(os.getcwd(), "_out"))

# Optional run date namespacing results.
RESULT_DATE = os.environ.get("T1D_RESULT_DATE", "").strip()

# Which prepared data set the experiments read.
DATA_DATE = os.environ.get("T1D_DATA_DATE", "").strip()


def _dated(sub: str, date: str) -> str:
    """``$ROOT/<sub>[/<date>]`` — nested under the date when non-empty."""
    d = os.path.join(OUTPUT_ROOT, sub)
    return os.path.join(d, date) if date else d


#: WHICH experiment is writing. Keeps one arm's tables from truncating another's.
EXPERIMENT = os.environ.get("T1D_EXPERIMENT", "").strip()


def _exp(base: str) -> str:
    """``<base>/<EXPERIMENT>`` when an experiment is named, else ``<base>``."""
    return os.path.join(base, EXPERIMENT) if EXPERIMENT else base


DATA_DIR = _dated("data", DATA_DATE)                # PREP outputs (dated by --data)
RESULTS_DIR = _exp(_dated("results", RESULT_DATE))  # per-run, per-EXPERIMENT results


# ---------------------------------------------------------------------------
# Prepared data
# ---------------------------------------------------------------------------
def train_trajectory_path(safe_name: str) -> str:
    """Per-patient training trajectory: ``data/<date>/train/<name>.npz``."""
    return os.path.join(DATA_DIR, "train", f"{safe_name}.npz")


def test_trajectory_path(safe_name: str) -> str:
    """Per-patient test set: ``data/<date>/test/<name>.npz``."""
    return os.path.join(DATA_DIR, "test", f"{safe_name}.npz")


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
def long_table_path() -> str:
    """The benchmark's long table: one row per (patient, day, therapy, method)."""
    return os.path.join(RESULTS_DIR, "benchmark_long.csv")


def decisions_table_path() -> str:
    """Per-(patient, day, method) decision metrics: ``benchmark_decisions.csv``."""
    return os.path.join(RESULTS_DIR, "benchmark_decisions.csv")


def benchmark_report_path() -> str:
    """Aggregate analysis: ``results/<date>/<experiment>/benchmark_report.md``."""
    return os.path.join(RESULTS_DIR, "benchmark_report.md")


def replication_ceiling_path() -> str:
    """Per-patient replication ceiling: ``replication_ceiling.csv``."""
    return os.path.join(RESULTS_DIR, "replication_ceiling.csv")


def replication_ceiling_days_path() -> str:
    """Per-(patient, day, seed) detail: ``replication_ceiling_days.csv``."""
    return os.path.join(RESULTS_DIR, "replication_ceiling_days.csv")
