# Evaluating Type-1 Diabetes Digital Twins by Decision Quality

[![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)

**Paper:** [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX)

Code for the evaluation protocol and benchmark in the paper. A digital twin (DT) for type 1 diabetes (T1D) is fit to a virtual patient's CGM under their baseline treatment, then generates glucose trajectories under 20 counterfactual insulin treatments it has never seen. A glycemic risk kernel scores each trajectory, which gives the DT's ranking of the candidate treatments. The benchmark scores the DT on **decision quality**, meaning whether that ranking agrees with the patient's true ranking, and reports it alongside **prediction accuracy**, meaning how closely the predicted glucose matches the true glucose.

Everything runs on the 30 virtual patients that ship with [`simglucose`](https://github.com/jxx123/simglucose) (UVA/Padova). There is no patient data in this repository. A strong score on this benchmark is not clinical validation, and no result here is treatment guidance.

---

## 1. Contents

```
experiments/
  patients.py                  the 30 virtual patients, the basal-bolus controller, and the simulator run loop
  scenario.py                  sample interval, scored window, candidate treatments, meals
  trajectories.py              DATA: generates and caches the training and evaluation .npz files
  scoring.py                   metric definitions and the two table writers
  run_benchmark.py             DRIVER: fit -> predict -> score
  merge_results.py             combines per-method result tables into one
  analyze_benchmark.py         Friedman -> Holm-corrected pairwise Wilcoxon -> benchmark_report.md
  plot_critical_difference.py  critical difference diagrams (the paper figure)
  sensitivity.py               leave-one-patient-out and leave-one-age-group-out sensitivity
  replication_ceiling.py       checks that the ground truth is deterministic
  output_paths.py              on-disk layout
  logsetup.py                  quiet/verbose control (quiet is the default)
requirements.txt
t1d_twin/                      the physiological model, the DT methods, and metric primitives
  replaybg_model.py            the nine-state, eight-parameter ReplayBG ODE, fixed-step RK4
                               state order [Isc1, Isc2, Ip, Qsto1, Qsto2, Qgut, G, X, IG]
  replaybg_priors.py           the published ReplayBG prior
  value.py                     the Magni risk kernel and the reward R
  sensor.py                    Breton-Kovatchev CGM noise
  units.py                     simglucose <-> ReplayBG unit conversions
  simglucose_adapter.py        RunResult, the container every simulator run is returned in
  copula.py                    multivariate t-copula over an MCMC chain
  twin.py                      the DT interface every method implements
  identify_mcmc.py             PUBLISHED METHOD: Markov chain Monte Carlo (MCMC), ReplayBG
  identify_sbi.py              PUBLISHED METHOD: simulation-based inference (SBI), neural posterior estimation
  identify_ukf.py              PUBLISHED METHOD: unscented Kalman filter (UKF), joint state-parameter filtering
  identify_linear.py           BASELINE: linear response model (Linear), ridge regression on lagged inputs
  identify_neural.py           BASELINE: neural response model (Neural), MLP on Linear's features
  identify_prior.py            BASELINE: average patient (Prior), replay over the ReplayBG prior, reads no CGM
```

---

## 2. Environment

Python 3.12. Everything runs on CPU — see [§8](#8-hardware).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## 3. Configuration

Five environment variables set the on-disk layout and logging. Nothing else is configurable by path, and no path is hard-coded.

| variable          | default  | meaning                                                                               |
| ----------------- | -------- | ------------------------------------------------------------------------------------- |
| `T1D_OUTPUT_ROOT` | `./_out` | root for everything written                                                           |
| `T1D_DATA_DATE`   | _(none)_ | namespaces the generated trajectories: `$T1D_OUTPUT_ROOT/data/<date>/`                |
| `T1D_RESULT_DATE` | _(none)_ | namespaces results: `$T1D_OUTPUT_ROOT/results/<date>/`                                |
| `T1D_EXPERIMENT`  | _(none)_ | namespaces results **within** a date: `$T1D_OUTPUT_ROOT/results/<date>/<experiment>/` |
| `T1D_VERBOSE`     | unset    | `1` turns on progress bars and per-method logging                                     |

The resulting tree:

```
$T1D_OUTPUT_ROOT/
  data/<data-date>/train/<patient>.npz     30 training days under the baseline treatment
  data/<data-date>/test/<patient>.npz      10 evaluation days x 20 candidate treatments
  results/<result-date>/<experiment>/
      benchmark_long.csv                     one row per (patient, day, treatment, method)
      benchmark_decisions.csv                one row per (patient, day, method)
      benchmark_report.md                    the statistical analysis
      critical_difference_diagram.{pdf,png}  the paper figure
      sensitivity_{report.md,pairs.csv,claims.csv}  the sensitivity analysis
      replication_ceiling{,_days}.csv        the determinism check
  sbi_logs/<tag>/                          TensorBoard traces from SBI training
```

---

## 4. Benchmark Design

| Component             | Choice                                                                                                  |
| --------------------- | ------------------------------------------------------------------------------------------------------- |
| Simulator             | `simglucose` (UVA/Padova), treated as a black box                                                       |
| Patients              | 30 virtual patients: 10 adults, 10 adolescents, 10 children                                             |
| CGM                   | Breton-Kovatchev noise, 5 min interval, 288 samples per 24 h day                                        |
| Meals                 | `simglucose`'s `RandomScenario`, one independent draw per day, shared by every treatment and method     |
| Baseline treatment π₀ | the patient's own basal rate, carbohydrate ratio and correction factor                                  |
| Candidate treatments  | 20 counterfactuals: meal bolus or basal rate scaled by −50% to +50% in 10% steps, the other unchanged   |
| Correction bolus      | computed from noise-free interstitial glucose (IG), so the true ranking is deterministic                |
| Treatment switch      | every run starts on π₀ at midnight and switches to the candidate at 05:00, when the scored window opens |
| Risk kernel           | reward R = negative total Magni risk over the scored 24 h window, glucose clipped to [40, 400] mg/dL    |
| Training data         | 30 independent identification days under π₀                                                             |
| Evaluation data       | 10 independent identification days × 20 candidate treatments                                            |

Every metric is computed per patient and evaluation day on noise-free IG:

| Family              | Metric      | Definition                                                                                  |
| ------------------- | ----------- | ------------------------------------------------------------------------------------------- |
| Prediction accuracy | RMSE        | root mean squared error between predicted and true IG, averaged over the 20 treatments      |
|                     | MARD        | mean absolute relative difference, averaged over the 20 treatments                          |
| Decision quality    | Spearman    | rank correlation between the DT's and the patient's ranking of the treatments (**primary**) |
|                     | Regret      | true reward of the true best treatment minus that of the DT's top treatment                 |
|                     | Regret Norm | Regret divided by the range of true rewards across the treatments                           |

---

## 5. Reproducing the Benchmark

The pipeline is five steps. Set the environment once per shell:

```bash
export T1D_OUTPUT_ROOT="$PWD/_out"
export T1D_DATA_DATE=2026-08-13
export T1D_RESULT_DATE=2026-08-24
```

### Step 1 — Generate the Data

```bash
T1D_EXPERIMENT=benchmark
python -m experiments.trajectories --jobs 30
```

30 patients × (30 training days + 10 evaluation days × 20 treatments). Each day is an independent 29 h run from steady state at midnight; the first 5 h are dropped and the 24 h from 05:00 are kept. Files are written atomically and cached, so a re-run with the same configuration is a no-op and `--force` rebuilds. A lock file refuses two concurrent jobs writing the same data date.

Useful flags: `--kinds adult,adolescent,child`, `--limit N`, `--train-days`, `--test-days`.

### Step 2 — Run the DT Methods

MCMC costs roughly 45× the other five methods combined, so run it on its own and merge. This is how the paper's results were produced.

```bash
# the five fast methods, one job, ~14 min wall at 30 workers
T1D_EXPERIMENT=benchmark
python -m experiments.run_benchmark --methods sbi,ukf,linear,neural,prior --jobs 30

# MCMC alone, ~8 h 30 m wall at 30 workers
T1D_EXPERIMENT=mcmc
python -m experiments.run_benchmark --methods mcmc --jobs 30
```

Each DT is fit once from its identification window and then generates the 10 × 20 evaluation trajectories. The driver exits non-zero if any patient or method raised, and prints which. `--traceback` keeps the full traceback. `--seed` (default 42) sets every method's seed.

Each method uses the amount of training data its original implementation prescribes. The identification windows are fixed in `run_benchmark.METHOD_FIT_HOURS` / `METHOD_FIT_DAYS` and are not command-line options:

| Method                    | Role             | Identification Window                    | Fit + Score (per patient, 1 core) |
| ------------------------- | ---------------- | ---------------------------------------- | --------------------------------: |
| `mcmc` — MCMC             | published method | one 22 h window (264 CGM samples)        |                          8 h 18 m |
| `sbi` — SBI               | published method | one 22 h window (264 CGM samples)        |                             245 s |
| `ukf` — UKF               | published method | one 22 h window (264 CGM samples)        |                             181 s |
| `prior` — Average patient | baseline         | none (reads no CGM)                      |                             175 s |
| `neural` — Neural         | baseline         | all 30 training days (8,640 CGM samples) |                             2.4 s |
| `linear` — Linear         | baseline         | all 30 training days (8,640 CGM samples) |                             0.1 s |

Costs are medians over the 30 patients and cover the fit _and_ all 200 predicted trajectories; see [§8](#8-hardware).

### Step 3 — Merge the Methods and Run the Statistical Analysis

```bash
T1D_EXPERIMENT=benchmark
python -m experiments.merge_results --sources benchmark,mcmc --into combined --analyze
```

`merge_results` combines the result tables into a third namespace. It refuses duplicate methods, mismatched `(patient, day)` grids and schema drift, and it never edits a source. `--sources` also takes `<date>/<namespace>` to pull a method from a different result date, or an absolute path. `--analyze` runs the analysis over the merged table, which is the only correct place to run it: the Holm correction is computed over the methods present, so analyzing methods separately and pasting the results together gives _p_-values that match no stated number of comparisons.

`T1D_EXPERIMENT` must be set for `merge_results` to resolve the date directory.

The statistical analysis follows Demšar (2006). Metrics are first averaged across evaluation days within each patient. An omnibus Friedman test runs per metric on the within-patient ranks of the methods, and, only where it rejects at α = 0.05, pairwise Wilcoxon signed-rank tests with a Holm correction identify which pairs differ.

To re-run the analysis alone over tables you already have:

```bash
python -m experiments.analyze_benchmark \
    --decisions _out/results/2026-08-24/combined/benchmark_decisions.csv \
    --long      _out/results/2026-08-24/combined/benchmark_long.csv \
    --out       _out/results/2026-08-24/combined/benchmark_report.md
```

### Step 4 — Critical Difference Diagrams

```bash
T1D_EXPERIMENT=combined
python -m experiments.plot_critical_difference
```

Draws one critical difference diagram per metric, RMSE on top and Spearman below, the primary metric of each family. Each method sits at its mean rank across patients (1 is best). A bar joins the methods that the Holm-corrected pairwise Wilcoxon tests cannot separate; a filled circle marks a member of the group, and an open circle marks a method the bar passes over but does not include. No Nemenyi critical-difference interval is drawn. The figure is written as `critical_difference_diagram.pdf` and `.png` beside the report. It imports the analysis module's tests rather than recomputing them, so the figure and the tables cannot disagree. `--metrics` picks the panels and `--width` defaults to 5.5 in.

`--methods` draws a subset and corrects Holm over exactly that subset. Use it whenever the merged table carries methods a figure should not show, or the figure's _p_-values will silently disagree with the tables.

### Step 5 — Sensitivity Analysis

```bash
T1D_EXPERIMENT=combined
python -m experiments.sensitivity
```

Repeats the entire statistical analysis with each patient left out in turn (30 repetitions on 29 patients), and again with each age group of 10 patients left out (3 repetitions on 20), and reports how often each comparison stated in the paper's results still holds. Writes `sensitivity_report.md`, `sensitivity_pairs.csv` and `sensitivity_claims.csv` beside the decisions table. `--decisions` points it at a different table.

---

## 6. Verifying the Ground Truth Is Deterministic

The correction bolus is computed from noise-free IG rather than from the CGM reading, so the delivered insulin, and therefore the patient's true ranking, is a deterministic property of the patient, the meals and the treatment, regardless of the sensor seed. `replication_ceiling` checks this: it re-runs the ground truth under fresh sensor seeds and scores each replicate against the cached evaluation data exactly as a DT is scored. A DT that reproduced the simulator exactly would score Spearman 1.0.

```bash
T1D_EXPERIMENT=ceiling
python -m experiments.replication_ceiling --jobs 30 --verify
```

Every check must pass:

```
max_spearman == 1        regret_norm_floor == 0
rmse_floor == 0          mard_floor == 0        reference re-run drift == 0
```

`--verify` additionally re-runs the reference seed and reports the maximum IG drift from the cached bytes. Anything but exactly `0.0` invalidates the whole dataset, and the driver exits 2.

---

## 7. Smoke Test (~1 minute)

Before a full run, test the pipeline end to end on one patient and one day:

```bash
export T1D_OUTPUT_ROOT=/tmp/t1d_smoke T1D_DATA_DATE=smoke T1D_RESULT_DATE=smoke

T1D_EXPERIMENT=benchmark python -m experiments.trajectories \
    --kinds adult --limit 1 --train-days 1 --test-days 1
T1D_EXPERIMENT=benchmark python -m experiments.run_benchmark \
    --kinds adult --limit 1 --train-days 1 --test-days 1 --methods linear
T1D_EXPERIMENT=prior     python -m experiments.run_benchmark \
    --kinds adult --limit 1 --train-days 1 --test-days 1 --methods prior
T1D_EXPERIMENT=benchmark python -m experiments.merge_results \
    --sources benchmark,prior --into combined --analyze
T1D_EXPERIMENT=ceiling   python -m experiments.replication_ceiling \
    --kinds adult --limit 1 --train-days 1 --test-days 1 --seeds 1 --verify
```

These reduced budgets only check that the pipeline runs. Never quote a number from one; a smoke-sized SBI training set in particular under-trains badly.

---

## 8. Hardware

CPU only. No step uses a GPU, and adding one does not help. Parallelism is across the 30 patients; SBI's flow and Neural's 9,793-parameter MLP are both small, and both pin torch to one thread so the workers do not oversubscribe each other.

`--jobs N` spawns N worker processes, one patient each. Wall clock scales with the number of waves, so 30 workers finish the cohort in one wave and 8 workers take four.

### What the Paper's Runs Used

An internal SLURM cluster of 64-core, 512 GB nodes. Each job requested 30 cores and 128 GB with one worker per patient, so the whole cohort is fit and scored concurrently.

### Measured Cost

Cost of fitting one patient and scoring all 20 candidate treatments on a **single core**, median over the 30 patients:

|                  |     MCMC |   SBI |   UKF | Prior | Neural | Linear |
| ---------------- | -------: | ----: | ----: | ----: | -----: | -----: |
| cost per patient | 8 h 18 m | 245 s | 181 s | 175 s |  2.4 s |  0.1 s |

By job:

| job                                                                                      | wall clock | core-hours |
| ---------------------------------------------------------------------------------------- | ---------: | ---------: |
| trajectory cache — 30 patients × (30 training days + 10 evaluation days × 20 treatments) |    1,182 s |        9.9 |
| the five fast methods, one job                                                           |      861 s |        5.5 |
| MCMC, separately                                                                         |   8 h 30 m |        249 |
| **total**                                                                                |            |    **264** |

---

## 9. Output Schema

`benchmark_long.csv` — one row per `(patient, day, treatment, method)`:

`patient, day, therapy, method, true_reward, pred_reward, true_rank, pred_rank, rmse, mard`

`benchmark_decisions.csv` — one row per `(patient, day, method)`; this is the table the analysis reads:

`patient, day, method, spearman, regret, regret_norm, rmse, mard, true_best, pred_best, n_therapies, degenerate, nonfinite`

The column names use `therapy` for a treatment. Decision quality metrics are `spearman` (primary), `regret` and `regret_norm`; prediction accuracy metrics are `rmse` and `mard`. Rank 1 is best. All metrics are computed on noise-free IG over the scored 24 h window.

---

## 10. Adding a DT Method

A DT method is anything that maps a patient's CGM, recorded under the baseline treatment, to a DT, and a DT is anything that maps the carbohydrate and insulin rates delivered under a treatment to a predicted glucose trajectory. Those are the only requirements. No states, parameter vector or physiological model are assumed, so mechanistic and black-box methods are scored identically.

Implement `t1d_twin.twin.Twin` (`predict_ig` and `summary`; `generate_cgm` comes from the base class) and expose an `identify_twin_from_run(run, dt=..., seed=...)` entry point. Then register the method in `experiments/run_benchmark.py`: `ALL_METHODS`, a `_fit_*` function with its branch in `_one`, `METHOD_COVARIATES`, `METHOD_FIT_HOURS`, and `METHOD_FIT_DAYS` if it uses more than one day. Finally, add it to `METHOD_ORDER` and `PRETTY_METHOD` in `experiments/analyze_benchmark.py` so the report and the figure label it.

---

## 11. Citing

If you use this code or benchmark, please cite our paper: [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX).
