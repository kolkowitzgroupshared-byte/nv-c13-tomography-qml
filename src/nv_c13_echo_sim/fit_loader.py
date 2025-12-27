from __future__ import annotations
import numpy as np
from typing import Any, Dict, List, Tuple

def build_param_matrix_from_fit(fit: Dict[str, Any]) -> Dict[str, Any]:
    keys = fit["unified_keys"]
    labels = list(map(int, fit["nv_labels"]))
    nv_orientations = [tuple(o) for o in fit["orientations"]]
    popts = fit["popts"]
    chis = fit.get("red_chi2", [None] * len(popts))

    def _asdict(p):
        d = {k: None for k in keys}
        if p is None:
            return d
        for k, v in zip(keys, p + [None] * (len(keys) - len(p))):
            d[k] = v
        return d

    P = np.full((len(popts), len(keys)), np.nan, float)
    for i, p in enumerate(popts):
        d = _asdict(p)
        for j, k in enumerate(keys):
            v = d[k]
            if v is None:
                continue
            try:
                P[i, j] = float(v)
            except Exception:
                pass

    K = {k: j for j, k in enumerate(keys)}
    return dict(
        keys=keys,
        labels=labels,
        nv_orientations=nv_orientations,
        popts=popts,
        chis=chis,
        P=P,
        K=K,
    )


def cohort_median_mad(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    def _nanmedian(x):
        m = np.nanmedian(x)
        return m if np.isfinite(m) else np.nan

    def _mad(x):
        med = _nanmedian(x)
        if not np.isfinite(med):
            return np.nan
        return _nanmedian(np.abs(x - med)) * 1.4826

    med = np.array([_nanmedian(P[:, j]) for j in range(P.shape[1])])
    mad = np.array([_mad(P[:, j]) for j in range(P.shape[1])])
    return med, mad
