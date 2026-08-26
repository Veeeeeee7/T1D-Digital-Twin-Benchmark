"""Unit conversions between simglucose I/O and ReplayBG model inputs."""
from __future__ import annotations

MG_PER_G = 1000.0   # mg per g
MU_PER_U = 1000.0   # mU per U


def insulin_U_per_min_to_mU_per_kg_min(u_per_min: float, BW: float) -> float:
    """simglucose insulin rate (U/min) -> ReplayBG I(t) (mU/kg/min)."""
    return u_per_min * MU_PER_U / BW


def cho_g_per_min_to_mg_per_kg_min(g_per_min: float, BW: float) -> float:
    """simglucose meal rate (g/min) -> ReplayBG CHO(t) (mg/kg/min)."""
    return g_per_min * MG_PER_G / BW


# Inverse conversions (useful when driving simglucose from ReplayBG-side values).
def mU_per_kg_min_to_insulin_U_per_min(mu_per_kg_min: float, BW: float) -> float:
    """Convert insulin rate from mU/kg/min to U/min."""
    return mu_per_kg_min * BW / MU_PER_U


def mg_per_kg_min_to_cho_g_per_min(mg_per_kg_min: float, BW: float) -> float:
    """Convert carbohydrate rate from mg/kg/min to g/min."""
    return mg_per_kg_min * BW / MG_PER_G
