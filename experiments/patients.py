"""Virtual patients and the plant run loop for the therapy-ranking benchmark."""
from __future__ import annotations

import os
import re
import hashlib
import datetime
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd
import pkg_resources

from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.env import T1DSimEnv
from simglucose.controller.base import Controller, Action

from t1d_twin import units
from t1d_twin.replaybg_model import THETA_NAMES
from t1d_twin.simglucose_adapter import RunResult
# scenario owns the cadence, and the cadence comes FROM the sensor row, so the sensor
# name lives there too.
from experiments import scenario as _SC

# --- simglucose parameter tables (read once) -------------------------------
_VP_FILE = pkg_resources.resource_filename("simglucose", "params/vpatient_params.csv")
_Q_FILE = pkg_resources.resource_filename("simglucose", "params/Quest.csv")
_VP = pd.read_csv(_VP_FILE)
_Q = pd.read_csv(_Q_FILE)


# ===========================================================================
# Parameter access + synthesis
# ===========================================================================
def list_patients(kind: str | None = None) -> list[str]:
    """All base patient names, optionally filtered by ``kind`` prefix (``"adult"``,
    ``"adolescent"``, ``"child"``).
    """
    names = list(_VP["Name"])
    return [n for n in names if kind is None or n.startswith(kind)]


def patient_params(name: str) -> pd.Series:
    """The raw UVA/Padova parameter row for a base patient (a pandas Series)."""
    row = _VP.loc[_VP["Name"] == name]
    if row.empty:
        raise KeyError(f"unknown patient {name!r}; see list_patients()")
    return row.squeeze().copy()


def quest_params(name: str) -> dict:
    """``{CR, CF, TDI}`` therapy constants for a base patient."""
    row = _Q.loc[_Q["Name"] == name]
    if row.empty:
        raise KeyError(f"{name!r} not in Quest.csv")
    r = row.squeeze()
    return {"CR": float(r["CR"]), "CF": float(r["CF"]), "TDI": float(r["TDI"])}


# ===========================================================================
# Carb-counting-error perturbation (mis-set carb ratio)
# ---------------------------------------------------------------------------
CARB_ERROR_ENABLED = False

DEFAULT_DOSE_FRACTIONS = (0.75, 0.85, 1.15, 1.25)


# Settable parameter names (everything but the Name label).
PARAM_NAMES = [c for c in _VP.columns if c != "Name"]


def make_patient(params: pd.Series, init_state=None, seed: int | None = None) -> T1DPatient:
    """Construct a ``T1DPatient`` from a parameter Series."""
    return T1DPatient(params, init_state=init_state, seed=seed)


# ===========================================================================
# Controller that works for patients not in Quest.csv
# ===========================================================================
#: Correction bolus in the PLANT.
CORRECTION_ENABLED = True

#: WHICH glucose signal the correction term reads.
CORRECTION_SOURCE = "ig"


class ExplicitBBController(Controller):
    """Basal-bolus controller with explicit CR / CF / u2ss / BW."""

    def __init__(self, CR: float, CF: float, u2ss: float, BW: float,
                 target: float = 140.0,
                 bolus_factor: float = 1.0, basal_factor: float = 1.0,
                 correction: bool | None = None,
                 correction_source: str | None = None,
                 switch_after: int = 0) -> None:
        self.CR = float(CR)
        self.CF = float(CF)
        self.u2ss = float(u2ss)
        self.BW = float(BW)
        self.target = float(target)
        self.bolus_factor = float(bolus_factor)
        self.basal_factor = float(basal_factor)
        self.switch_after = int(switch_after)
        self._step = 0
        self.correction = (CORRECTION_ENABLED if correction is None
                           else bool(correction))
        src = (CORRECTION_SOURCE if correction_source is None
               else str(correction_source)).lower()
        if src not in ("ig", "cgm"):
            raise ValueError(f"correction_source must be 'ig' or 'cgm', got {src!r}")
        self.correction_source = src

    def policy(self, observation, reward, done, **kwargs):
        """Return the insulin action for one control step."""
        sample_time = kwargs.get("sample_time", 1)
        meal = kwargs.get("meal", 0)          # g/min
        # The signal the correction reads.
        if self.correction_source == "ig":
            if "bg" not in kwargs:
                raise KeyError(
                    "correction_source='ig' needs the env's noise-free `bg` in "
                    "kwargs; the caller must pass **info from T1DSimEnv")
            glucose = float(kwargs["bg"])
        else:
            glucose = float(observation.CGM)
        # Effective factors for THIS step.
        if self._step < self.switch_after:
            bolus_factor, basal_factor = 1.0, 1.0
        else:
            bolus_factor, basal_factor = self.bolus_factor, self.basal_factor
        self._step += 1

        basal = self.u2ss * self.BW / 6000.0  # U/min
        if meal > 0:
            # The carb term is a deterministic function of the meal plan and the
            # therapy.
            carb = (meal * sample_time) / self.CR
            bolus = carb
            if self.correction:
                bolus = bolus + (glucose > 150) * (glucose - self.target) / self.CF
            bolus = bolus / sample_time       # U -> U/min
            carb = carb / sample_time
        else:
            bolus = carb = 0.0
        # The carb-only bolus this same call produced, in U/min.
        self._last_bolus_nocorr = carb * bolus_factor
        return Action(basal=basal * basal_factor,
                      bolus=bolus * bolus_factor)

    def reset(self):
        """Clear per-run controller state between simulations."""
        self._last_bolus_nocorr = 0.0
        # MUST zero the therapy-switch counter: build_test reuses one controller
        # object for all 10 test days.
        self._step = 0


# ===========================================================================
# Run loop (drives a custom patient + controller through a simglucose env)
# ===========================================================================
def run_on_patient(params: pd.Series, controller: Controller, scenario, hours: float,
                   sensor_name: str | None = None, pump_name: str = "Insulet",
                   sensor_seed: int = 1, init_state=None) -> RunResult:
    """Drive ``controller`` on the patient ``params`` -> ``RunResult``."""
    if sensor_name is None:
        sensor_name = _SC.SENSOR_NAME
    patient = make_patient(params, init_state=init_state)
    BW = float(patient._params.BW)
    sensor = CGMSensor.withName(sensor_name, seed=sensor_seed)
    pump = InsulinPump.withName(pump_name)
    env = T1DSimEnv(patient, sensor, pump, scenario)
    sample_time = float(env.sample_time)

    if hasattr(controller, "reset"):
        controller.reset()
    obs, reward, done, info = env.reset()
    start_time = info["time"]

    n_steps = int(round(hours * 60.0 / sample_time))
    rows = []
    truncated = False
    for _ in range(n_steps):
        action = controller.policy(obs, reward, done, **info)
        obs, reward, done, info = env.step(action)

        cho_g_min = float(info["meal"])
        basal = float(action.basal)
        bolus = float(action.bolus)
        # Carb-only bolus from the SAME policy call that produced `action`, so it is
        # aligned with `bolus` rather than with the recorded CHO.
        bolus_nocorr = float(getattr(controller, "_last_bolus_nocorr", bolus))
        insulin_U_min = basal + bolus
        insulin_nocorr_U_min = basal + bolus_nocorr
        t_min = (info["time"] - start_time).total_seconds() / 60.0
        rows.append({
            "t_min": t_min, "datetime": info["time"],
            "CGM": float(obs.CGM), "BG": float(info["bg"]),
            "CHO_g_min": cho_g_min, "basal_U_min": basal, "bolus_U_min": bolus,
            "bolus_nocorr_U_min": bolus_nocorr,
            "insulin_U_min": insulin_U_min,
            "insulin_nocorr_U_min": insulin_nocorr_U_min,
            "CHO_mg_kg_min": units.cho_g_per_min_to_mg_per_kg_min(cho_g_min, BW),
            "insulin_mU_kg_min": units.insulin_U_per_min_to_mU_per_kg_min(insulin_U_min, BW),
            "insulin_nocorr_mU_kg_min": units.insulin_U_per_min_to_mU_per_kg_min(
                insulin_nocorr_U_min, BW),
            "lbgi": float(info["lbgi"]), "hbgi": float(info["hbgi"]),
            "risk": float(info["risk"]),
        })
        if done:
            truncated = True
            break

    # Worst-case padding for early-terminated episodes.
    completed_steps = len(rows)
    if truncated and completed_steps < n_steps and rows:
        last = rows[-1]
        for k in range(1, n_steps - completed_steps + 1):
            pad = dict(last)
            pad["t_min"] = last["t_min"] + k * sample_time
            pad["datetime"] = last["datetime"] + datetime.timedelta(minutes=k * sample_time)
            pad["CHO_g_min"] = 0.0
            pad["basal_U_min"] = 0.0
            pad["bolus_U_min"] = 0.0
            pad["insulin_U_min"] = 0.0
            pad["CHO_mg_kg_min"] = 0.0
            pad["insulin_mU_kg_min"] = 0.0
            rows.append(pad)

    df = pd.DataFrame(rows)
    return RunResult(df=df, BW=BW, sample_time=sample_time,
                     patient_name=str(params["Name"]), sensor_name=sensor_name,
                     sensor_seed=sensor_seed,
                     meta={"hours": hours, "start_time": start_time,
                           "synthetic": True, "n_steps": n_steps,
                           "completed_steps": completed_steps, "truncated": truncated})


# ===========================================================================
# Subject abstraction + patients.csv I/O
# ===========================================================================
_QUEST_COLS = ("CR", "CF", "TDI")

# Cached best-fit ReplayBG parameters (written by derive_replaybg_params.py and read
# back here so Phase 0/1/2 reuse the fit instead of recomputing it).
RBG_THETA_COLS = [f"rbg_{n}" for n in THETA_NAMES]
RBG_EXTRA_COLS = ["rbg_Ib", "rbg_CR_true", "rbg_fit_rmse"]
RBG_ALL_COLS = RBG_THETA_COLS + RBG_EXTRA_COLS


@dataclass
class Subject:
    """A (synthetic or base) virtual patient: UVA/Padova params + therapy constants."""
    name: str
    params: "pd.Series"
    quest: dict
    members: tuple = field(default_factory=tuple)
    cr_true: float | None = None
    dose_mult: float = 1.0
    # Cached best-fit ReplayBG parameters (None until derive_replaybg_params is run).
    rbg_theta: "np.ndarray | None" = None
    rbg_Ib: float | None = None
    rbg_cr_true: float | None = None
    rbg_fit_rmse: float | None = None

    def __post_init__(self):
        if self.cr_true is None:
            self.cr_true = float(self.quest["CR"])

    @property
    def cr_prog(self) -> float:
        """The programmed carb ratio actually used for dosing (= quest['CR'])."""
        return float(self.quest["CR"])

    @property
    def safe_name(self) -> str:
        """Filesystem-safe identifier for per-subject artifact/result folders."""
        return re.sub(r"[^A-Za-z0-9._+-]", "_", self.name)

    def baseline_controller(self, target: float = 140.0) -> "ExplicitBBController":
        """Controller implementing this subject's own carb-ratio therapy."""
        return ExplicitBBController(self.quest["CR"], self.quest["CF"],
                                    float(self.params["u2ss"]),
                                    float(self.params["BW"]), target=target)

    def therapy_controllers(self, bolus_factors=(0.85, 1.0, 1.5, 2.0, 2.5),
                            basal_factors=(), target: float = 140.0,
                            bolus_names=None, switch_after=None) -> dict:
        """Candidate controllers keyed by therapy name."""
        u2ss = float(self.params["u2ss"])
        BW = float(self.params["BW"])
        CR, CF = self.quest["CR"], self.quest["CF"]
        sw = _SC.therapy_switch_step() if switch_after is None else int(switch_after)
        if bolus_names is None:
            bolus_names = [f"bolus_x{f:.2f}" for f in bolus_factors]
        d = {}
        for f, nm in zip(bolus_factors, bolus_names):
            d[nm] = ExplicitBBController(CR, CF, u2ss, BW, target, bolus_factor=f,
                                         switch_after=sw)
        for f in basal_factors:
            d[f"basal_x{f:.2f}"] = ExplicitBBController(CR, CF, u2ss, BW, target,
                                                        basal_factor=f,
                                                        switch_after=sw)
        return d

    def run(self, controller, scenario, hours, sensor_seed: int = 1) -> RunResult:
        """Simulate this subject under a controller and scenario."""
        return run_on_patient(self.params, controller, scenario, hours,
                              sensor_seed=sensor_seed)


#: The benchmark's subject set: every virtual patient simglucose ships.
DEFAULT_KINDS = ("adult", "adolescent", "child")


def benchmark_subjects(kinds=DEFAULT_KINDS) -> list["Subject"]:
    """The subject set every run uses, in a fixed order."""
    names = [n for k in kinds for n in list_patients(k)]
    return [subject_from_base(n) for n in names]


def subject_from_base(name: str) -> Subject:
    """Wrap a built-in simglucose patient as a Subject (uses its real CR/CF)."""
    return Subject(name=name, params=patient_params(name),
                   quest=quest_params(name), members=(name,))
