from __future__ import annotations
import numpy as np
from typing import Dict, Any

DEFAULT_BOUNDS = {
    "baseline": (0.0, 1.2),
    "comb_contrast": (0.0, 1.1),
    "revival_time_us": (25.0, 55.0),
    "width0_us": (1.0, 25.0),
    "T2_ms": (1e-3, 2e3),
    "T2_exp": (0.6, 4.0),
    "amp_taper_alpha": (0.0, 2.0),
    "width_slope": (-0.2, 0.2),
    "revival_chirp": (-0.06, 0.06),
    "osc_amp": (-0.3, 0.3),
    "osc_f0": (0.0, 0.50),
    "osc_f1": (0.0, 0.50),
    "osc_phi0": (-np.pi, np.pi),
    "osc_phi1": (-np.pi, np.pi),
}

def _clip(bounds: Dict[str, Any], k: str, v: float) -> float:
    lo, hi = bounds[k]
    return float(np.minimum(hi, np.maximum(lo, v)))

def nv_prior_draw(
    *,
    nv_idx: int,
    P: np.ndarray,
    K: Dict[str, int],
    cohort_med: np.ndarray,
    cohort_mad: np.ndarray,
    rng: np.random.Generator,
    mix_global: float = 0.3,
    bounds: Dict[str, Any] = DEFAULT_BOUNDS,
) -> Dict[str, float]:
    out = {}
    for k, j in K.items():
        mu = P[nv_idx, j] if np.isfinite(P[nv_idx, j]) else cohort_med[j]

        if not np.isfinite(mu):
            # same fallbacks you used
            if k == "baseline":
                mu = 0.6
            elif k == "comb_contrast":
                mu = 0.45
            elif k == "revival_time_us":
                mu = 38.0
            elif k == "width0_us":
                mu = 7.0
            elif k == "T2_ms":
                mu = 0.08
            elif k == "T2_exp":
                mu = 1.2
            else:
                mu = 0.0

        sig = cohort_mad[j]
        if (not np.isfinite(sig)) or sig == 0.0:
            sig = 1e-3

        widen = 1.0 if k != "osc_amp" else 1.5
        draw = mu + widen * mix_global * sig * rng.standard_normal()
        out[k] = _clip(bounds, k, float(draw)) if k in bounds else float(draw)

    return out
