"""Cadence, scored window, candidate therapy grid, and meal scenarios."""
from __future__ import annotations

import numpy as np
from datetime import datetime

from simglucose.simulation.scenario import CustomScenario

# --- cadence / horizon ------------------------------------------------------
#: CGM cadence [min].
SAMPLE_TIME = 5.0

#: simglucose's 5-minute sensor row.
SENSOR_NAME = "GuardianRT"

DT = 1.0                 # ODE integration grid [min] (simglucose T1DPatient.SAMPLE_TIME)
WINDOW_HOURS = 24.0      # the identification / scoring WINDOW — 05:00 to 05:00
N_OBS = int(WINDOW_HOURS * 60 / SAMPLE_TIME)     # 288 CGM samples per window

#: Hour of run at which the scored window opens; the hours before it are burn-in.
WINDOW_START_H = 5.0
RUN_HOURS = WINDOW_START_H + WINDOW_HOURS        # 29 h simulated per day
N_RUN = int(RUN_HOURS * 60 / SAMPLE_TIME)        # 348 CGM samples simulated
WINDOW_OFFSET = int(WINDOW_START_H * 60 / SAMPLE_TIME)   # 60 samples discarded

#: Large enough that run ``day``'s second calendar day can never be run ``day+1``'s
#: first.
MIDNIGHT_STRIDE = 100000

#: WHEN a candidate therapy starts, as an hour-of-run.
THERAPY_START_H = WINDOW_START_H


def therapy_switch_step(sample_time: float = SAMPLE_TIME) -> int:
    """Controller steps spent on the BASELINE therapy before the candidate starts."""
    return int(round(THERAPY_START_H * 60.0 / float(sample_time)))


#: The identification window **ReplayBG and SBI publish**: 22 h at 5 min = 264
#: samples, exactly Hoang's ``y in R^264`` and exactly the span of.
PAPER_FIT_HOURS = 22.0
N_PAPER_FIT = int(PAPER_FIT_HOURS * 60 / SAMPLE_TIME)  # 264

START = datetime(2024, 1, 1, 0, 0, 0)            # midnight, fasting, steady state
SEED = 1

# --- meals ------------------------------------------------------------------

#: The meal distribution is simglucose's, not ours.
MEAL_SOURCE = "simglucose.simulation.scenario_gen.RandomScenario"

# --- candidate therapies Pi (|Pi| = 20) -------------------------------------
REL_BOLUS = (0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5)

#: Basal multipliers: ReplayBG Scenario 3, same range and step, minus 1.0.
BASAL_FACTORS = (0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5)

#: The therapy the training trajectory is generated under, and the point both grids
#: are centred on.
TRAINING_FACTOR = 1.0

N_THERAPIES = len(REL_BOLUS) + len(BASAL_FACTORS)            # 20


def day_seed(seed: int = SEED, day: int = 0) -> int:
    """Deterministic per-day seed for ``RandomScenario``."""
    return int(np.random.SeedSequence([int(seed), int(day)]).generate_state(1)[0])


def _one_day_meals(seed: int, day: int):
    """One calendar day's draw from simglucose, ``[(min_since_midnight, g), ...]``."""
    from simglucose.simulation.scenario_gen import RandomScenario
    sc = RandomScenario(start_time=START, seed=day_seed(seed, day)).scenario["meal"]
    return sorted((float(t), float(a)) for t, a in zip(sc["time"], sc["amount"])
                  if a > 0)


def meal_plan(hours: float = RUN_HOURS, seed: int = SEED, day: int = 0):
    """Meals for one independent run, ``[(minutes_from_start, grams), ...]``."""
    out = list(_one_day_meals(seed, day))
    n_midnights = int(np.ceil(hours / 24.0)) - 1
    for k in range(1, n_midnights + 1):
        for t, a in _one_day_meals(seed, day + k * MIDNIGHT_STRIDE):
            out.append((t + k * 1440.0, a))
    return sorted((t, a) for t, a in out if t < hours * 60.0)


def make_scenario(hours: float = RUN_HOURS, seed: int = SEED, day: int = 0):
    """A simglucose ``CustomScenario`` for one independent run."""
    meals = meal_plan(hours, seed, day)
    return CustomScenario(start_time=START,
                          scenario=[(m / 60.0, g) for m, g in meals]), meals


def window_slice(n_per_sample: int = 1):
    """``slice`` selecting the scored window out of a full ``RUN_HOURS`` run."""
    r = int(n_per_sample)
    return slice(WINDOW_OFFSET * r, (WINDOW_OFFSET + N_OBS) * r)


#: Day indices are namespaced so a training day and a test day can never be the same
#: meal realization.
TRAIN_DAY0 = 0        # training days: 0 .. TRAIN_DAYS-1
TEST_DAY0 = 1000      # test days:  1000 .. 1000+TEST_DAYS-1

TRAIN_DAYS = 30
TEST_DAYS = 10


def relative_factors(dose_mult_m: float):
    """``(bolus_factors, bolus_names)`` for this patient's relative grid."""
    factors = tuple(float(dose_mult_m) * r for r in REL_BOLUS)
    names = tuple(f"bolus_r{r:.2f}" for r in REL_BOLUS)
    return factors, names


def therapy_names(dose_mult_m: float | None = None):
    """The ordered policy names. Order fixes the column order of every metric."""
    return ([f"bolus_r{r:.2f}" for r in REL_BOLUS]
            + [f"basal_x{f:.2f}" for f in BASAL_FACTORS])


def baseline_relative_position(d: float) -> float:
    """Where the patient's own therapy sits on the relative grid: ``r = d``."""
    return float(d)
