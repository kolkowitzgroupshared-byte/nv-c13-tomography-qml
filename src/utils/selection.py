from __future__ import annotations
import numpy as np
from typing import List, Optional

def _to_float_or_inf(x):
    try:
        if x is None:
            return np.inf
        v = float(x)
        return v if np.isfinite(v) else np.inf
    except Exception:
        return np.inf

def select_nvs(
    *,
    P: np.ndarray,
    K: dict,
    popts: list,
    chis: list,
    max_keep: int = 15,
    contrast_min: float = 0.10,
    contrast_max: float = 2.00,
    t2_threshold_us: float = 2000.0,
) -> List[int]:
    k_cc = K["comb_contrast"]
    k_T2 = K["T2_ms"]

    cc = P[:, k_cc]
    has_cc = np.isfinite(cc)
    meets_cc = has_cc & (np.abs(cc) >= contrast_min)
    if contrast_max is not None and np.isfinite(contrast_max):
        meets_cc &= np.abs(cc) <= contrast_max

    T2_ms = P[:, k_T2]
    T2_us = T2_ms * 1000.0
    fast_T2 = np.isfinite(T2_us) & (T2_us < t2_threshold_us)

    chi_vals = np.array([_to_float_or_inf(c) for c in chis], float)
    order = np.argsort(chi_vals)

    keep = [
        i
        for i in order
        if (popts[i] is not None and np.isfinite(chi_vals[i]) and fast_T2[i] and meets_cc[i])
    ][:max_keep]

    return keep
