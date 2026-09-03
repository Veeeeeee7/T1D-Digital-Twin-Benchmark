# Evaluating Type-1 Diabetes Digital Twins by Decision Transfer

Code to reproduce the benchmark. Fits a digital twin to a virtual patient's CGM under one insulin treatment, ask it to predict glucose under 20 treatments it has never seen, and score it on whether it ranks those treatments the way the real patient would, not only on how closely it tracks glucose.

Everything runs on the 30 virtual subjects distributed with [`simglucose`](https://github.com/jxx123/simglucose) (UVA/Padova). There is no patient data anywhere in this repository, and no component issues a dose recommendation.

---

## 1. Contents

```
experiments/
  patients.py             the 30 built-in subjects, the basal-bolus controller, and the plant run loop
  scenario.py             cadence, scored window, candidate treatment grid, meals
  trajectories.py         DATA: generates and ships the train/test .npz files
  scoring.py              metric definitions and the two table writers
  run_benchmark.py        DRIVER: fit -> predict -> score
  merge_results.py        unions per-experiment result tables into one
  analyze_benchmark.py    Friedman -> Holm-corrected Wilcoxon -> benchmark_report.md
  plot_critical_difference.py  the same post-hoc drawn as a critical difference diagram (the paper figure)
  replication_ceiling.py  determinism check
  output_paths.py         on-disk layout
  logsetup.py             quiet/verbose control (quiet is the default)
requirements.txt
t1d_twin/                 physics, twinning methods, and metric primitives
  replaybg_model.py       9-state ReplayBG ODE, fixed-step RK4 state order [Isc1, Isc2, Ip, Qsto1, Qsto2, Qgut, G, X, IG]
  replaybg_priors.py      the published Cappon prior; numpy sampler + torch wrapper
  value.py                Magni risk kernel and the reward R
  sensor.py               UVA/Padova (Breton-Kovatchev) CGM noise
  units.py                simglucose <-> ReplayBG unit conversions
  simglucose_adapter.py   RunResult, the container every run is returned in
  copula.py               multivariate t-copula over an MCMC chain
  twin.py                 the Twin interface every method implements
  identify_mcmc.py        METHOD: MCMC (ReplayBG)
  identify_sbi.py         METHOD: SBI / neural posterior estimation
  identify_ukf.py         METHOD: joint state-parameter unscented Kalman filter
  identify_linear.py      BASELINE: ridge finite-impulse-response regression
  identify_neural.py      BASELINE: MLP over identify_linear's design matrix
  identify_prior.py       BASELINE: ReplayBG replay over prior draws, reads no CGM
```

---

## 2. Environment

Python 3.12 and everything is CPU-only — see [§7](#7-hardware).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## 3. Configuration

Four environment variables set the on-disk layout. Nothing else is configurable by path, and no path is hard-coded.

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
  data/<data-date>/train/<patient>.npz     30 identification days, training treatment
  data/<data-date>/test/<patient>.npz      10 days x 20 candidate treatments
  results/<result-date>/<experiment>/
      benchmark_long.csv                   one row per (patient, day, treatment, method)
      benchmark_decisions.csv              one row per (patient, day, method)
      benchmark_report.md                  the analysis
      critical_difference_diagram.{pdf,png}  the paper figure
      replication_ceiling{,_days}.csv      the determinism check
  sbi_logs/<tag>/                          TensorBoard traces from SBI training
```

---

## 4. Reproducing the Benchmark

The pipeline is four steps. Set the environment once per shell:

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

30 subjects × (30 training days + 10 test days × 20 treatments), each an independent 29h run from steady state windowed 05:00 → 05:00. Files are written atomically and cached. A re-run with the same fingerprint is a no-op, and `--force` rebuilds. A lock file refuses two concurrent jobs writing the same data date.

Useful flags: `--kinds adult,adolescent,child`, `--limit N`, `--train-days`, `--test-days`.

### Step 2 — Run the Methods

`mcmc` costs roughly 45× the other five experiments combined, so run it on its own and merge. This is exactly how the reported runs were produced.

```bash
# the five fast arms, one job, ~14 min wall at 30 workers
T1D_EXPERIMENT=benchmark
python -m experiments.run_benchmark --methods sbi,ukf,linear,neural,prior --jobs 30

# mcmc alone, ~8 h 30 m wall at 30 workers
T1D_EXPERIMENT=mcmc
python -m experiments.run_benchmark --methods mcmc --jobs 30
```

Each patient is fit from a single window of training day 0, then predicts all 10 × 20 test cells. The driver exits non-zero if any patient or method raised, and prints which. `--traceback` keeps the full traceback. `--seed` (default 42) sets every method's seed.

Per-method identification budgets are fixed and are not command-line options. They are declared in `run_benchmark.METHOD_FIT_HOURS` / `METHOD_FIT_DAYS` so a method gets exactly what its own paper documents:

| Experiment | Role             | Identification Window                | Fit + Score (per patient, 1 core) |
| ---------- | ---------------- | ------------------------------------ | --------------------------------: |
| `mcmc`     | published method | one 22 h window (264 CGM samples)    |                          8 h 18 m |
| `sbi`      | published method | one 22 h window (264 samples)        |                             245 s |
| `ukf`      | published method | one 22 h window (264 samples)        |                             181 s |
| `prior`    | control          | none                                 |                             175 s |
| `neural`   | baseline         | all 30 training days (8,640 samples) |                             2.4 s |
| `linear`   | baseline         | all 30 training days (8,640 samples) |                             0.1 s |

Costs are medians over the 30 subjects and cover the fit _and_ all 200 test-cell predictions; see [§7](#7-hardware).

### Step 3 — Merge the Arms and Analyze

```bash
T1D_EXPERIMENT=benchmark
python -m experiments.merge_results --sources benchmark,mcmc --into combined --analyze
```

`merge_results` unions the namespaces into a third one, refusing duplicate arms, ragged `(patient, day)` grids and schema drift, and never editing a source. `--sources` also takes `<date>/<namespace>` to pull an arm from a different result date, or an absolute path. `--analyze` runs the analysis over the merged table, which is the only correct place to run it: the Holm correction is computed over the roster present, so analyzing arms separately and pasting the results together produces _p_-values that do not match any stated multiplicity burden.

`T1D_EXPERIMENT` must be set for `merge_results` to resolve the date directory. It derives the results root from the namespaced path.

The analysis is the Demšar (2006) protocol: an omnibus Friedman test per metric on within-patient ranks, then post-hoc pairwise Wilcoxon signed-rank with Holm correction, only where Friedman rejects. The unit of analysis is the patient with test days averaged within patient first, because patient difficulty dominates the between-method effect.

To re-run the analysis alone over tables you already have:

```bash
python -m experiments.analyze_benchmark \
    --decisions _out/results/2026-08-24/combined/benchmark_decisions.csv \
    --long      _out/results/2026-08-24/combined/benchmark_long.csv \
    --out       _out/results/2026-08-24/combined/benchmark_report.md
```

### Step 4 — Figures

```bash
T1D_EXPERIMENT=combined
python -m experiments.plot_critical_difference --metrics spearman,rmse
```

Draws one critical difference diagram per metric (Demšar 2006, in the Wilcoxon–Holm form: a bar joins every group of methods the Holm-corrected pairwise tests cannot separate, with no Nemenyi critical-difference interval) and writes it as `critical_difference_diagram.pdf` and `.png` beside the report. It imports the analysis module's post-hoc helpers rather than recomputing them, so the figure and the tables cannot disagree. `--width` defaults to 5.5 in.

`--methods` draws a subset and corrects Holm over exactly that subset. Use it whenever the merged vintage carries arms a paper does not report, or the figure's _p_-values will silently disagree with the tables.

---

## 5. Verifying the Plant is Deterministic

The correction bolus fires on noise-free interstitial glucose rather than on the CGM reading, which makes the plant deterministic given its seed and the true ranking a property of the patient and meals alone. `replication_ceiling` proves it: it re-runs the ground truth under fresh sensor seeds and scores each replicate against the shipped test set exactly as a method is scored. A perfect twin should therefore score exactly 1.0.

```bash
T1D_EXPERIMENT=ceiling
python -m experiments.replication_ceiling --jobs 30 --verify
```

Every check must pass:

```
max_spearman == 1        regret_norm_floor == 0
rmse_floor == 0          mard_floor == 0        reference re-run drift == 0
```

`--verify` additionally re-runs the reference seed and reports the maximum IG drift from the shipped bytes. Anything but exactly `0.0` voids the whole dataset, and the driver exits 2.

---

## 6. Smoke Test (~1 minute)

Before committing to a full run, test every experiment end to end on one patient, one day:

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

Reduced budgets are a plumbing check. Never quote a number from one — a smoke-sized SBI budget in particular under-trains badly.

---

## 7. Hardware

CPU only. No step uses a GPU, and adding one does not help. Parallelism is across the 30 patients, `sbi`'s flow and `neural`'s 9,793-parameter MLP are both tiny, and both pin torch to one intra-op thread so the workers do not oversubscribe each other.

`--jobs N` spawns N worker processes, one patient each. Scaling is linear in the number of patients per wave, so 30 workers finishes the cohort in one wave and 8 workers takes four.

### What the Reported Runs Ssed

An internal SLURM cluster of 64-core, 512 GB nodes. Each job requested 30 cores and 128 GB, one worker per patient, so the whole cohort is fit and scored concurrently.

### Measured cost

Per-patient fit-and-score cost on a **single core**, median over the 30 subjects:

|                  |     MCMC |   SBI |   UKF | Prior | Neural | Linear |
| ---------------- | -------: | ----: | ----: | ----: | -----: | -----: |
| cost per patient | 8 h 18 m | 245 s | 181 s | 175 s |  2.4 s |  0.1 s |

Aggregated by jobs:

| job                                                                                | wall clock | core-hours |
| ---------------------------------------------------------------------------------- | ---------: | ---------: |
| trajectory cache — 30 patients × (30 training days + 10 test days × 20 treatments) |    1,182 s |        9.9 |
| the five fast arms, one job                                                        |      861 s |        5.5 |
| `mcmc`, separately                                                                 |   8 h 30 m |        249 |
| **total**                                                                          |            |    **264** |

---

## 8. Output schema

`benchmark_long.csv` — one row per `(patient, day, treatment, method)`:

`patient, day, therapy, method, true_reward, pred_reward, true_rank, pred_rank, rmse, mard`

`benchmark_decisions.csv` — one row per `(patient, day, method)` (this is the table the analysis reads):

`patient, day, method, spearman, regret, regret_norm, rmse, mard, true_best, pred_best, n_therapies, degenerate, nonfinite`

Decision-quality metrics are `spearman` (primary), `regret` and `regret_norm`. Prediction accuracy is `rmse` and `mard`. Rank 1 is best. All metrics are computed on noise-free interstitial glucose over the scored 24 h window, with glucose clipped to [40, 400] mg/dL before the risk kernel is applied.

---

## 9. Adding a method

A twinning method is any input-conditioned predictor that can answer _what would this patient's glucose be under treatment π?_ for arbitrary π. That is the only admission requirement. No parameter vector, no states and no physiological model are assumed. Implement `t1d_twin.twin.Twin` (`predict_ig` and `summary`; `generate_cgm` comes from the base class), expose an `identify_twin_from_run(run, dt=..., seed=...)` entry point, then register the arm in four places in `experiments/run_benchmark.py`: `ALL_METHODS`, a `_fit_*` function with its branch in `_one`, `METHOD_COVARIATES`, `METHOD_FIT_HOURS` (and `METHOD_FIT_DAYS` if it consumes more than one day). Lastly, add it to `METHOD_ORDER` and
`PRETTY_METHOD` in `experiments/analyze_benchmark.py` so the report and the figure label it.

---

## 10. Citing

If you use this benchmark, please cite the paper. The methods it evaluates are reimplementations of published work and should be cited alongside it:

- **ReplayBG / MCMC** — Cappon et al., _ReplayBG: A Digital Twin-Based Methodology to Identify a Personalized Model from Type 1 Diabetes Data and Simulate Glucose Concentrations to Assess Alternative Therapies_, IEEE TBME, 2023.
- **SBI** — Hoang et al., _Real-Time Digital Twins for Type 1 Diabetes with Simulation-Based Inference_, arXiv:2507.01740, 2025.
- **UKF** — Hajizadeh et al., _Plasma Insulin Estimation in People with Type 1 Diabetes Mellitus_, 2018. The published method estimates plasma insulin on Hovorka's model and is validated against intravenous assays; this implementation retargets it to glucose on the ReplayBG model and is not a reproduction of that result.
- **Simulator** — Dalla Man et al., _The UVA/PADOVA Type 1 Diabetes Simulator: New Features_, 2014, via Xie's `simglucose`.
- **Risk kernel** — Magni et al., _Model Predictive Control of Type 1 Diabetes: An in Silico Trial_, 2007.
- **Statistical protocol** — Demšar, _Statistical Comparisons of Classifiers over Multiple Data Sets_, JMLR, 2006.

None of the six methods evaluated here is validated for clinical use, and no result in this repository is treatment guidance.
