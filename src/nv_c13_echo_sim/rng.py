from __future__ import annotations
import numpy as np
from typing import List, Optional

def spawn_streams(seed: Optional[int], num_streams: int, run_salt: Optional[int] = None) -> List[np.random.Generator]:
    if seed is None:
        ss = np.random.SeedSequence()
    else:
        if run_salt is None:
            ss = np.random.SeedSequence(int(seed))
        else:
            ss = np.random.SeedSequence(int(seed), spawn_key=[int(run_salt) & 0xFFFFFFFF])
    return [np.random.default_rng(cs) for cs in ss.spawn(num_streams)]

def _choose_sites(rng, present_sites, num_spins, selection_mode="uniform"):
    present_sites = list(present_sites)
    if num_spins is None or num_spins >= len(present_sites):
        return present_sites

    if selection_mode == "top_Apar":
        order = np.argsort([-abs(s["Apar_Hz"]) for s in present_sites])
        idx = order[:num_spins]

    elif selection_mode == "distance_weighted":
        r = np.array([max(s["dist"], 1e-12) for s in present_sites], float)
        w = 1.0 / (r**3)
        w /= w.sum()
        idx = np.sort(
            rng.choice(len(present_sites), size=num_spins, replace=False, p=w)
        )

    # NEW: pick strongest catalog entries by κ
    elif selection_mode == "top_kappa":
        order = np.argsort([-abs(s.get("kappa", 0.0)) for s in present_sites])
        idx = order[:num_spins]

    # NEW: pick by first-order line weight (minus+plus combined)
    elif selection_mode == "top_weight":
        order = np.argsort(
            [
                -(s.get("line_w_minus", 0.0) + s.get("line_w_plus", 0.0))
                for s in present_sites
            ]
        )
        idx = order[:num_spins]

    else:  # uniform
        idx = np.sort(rng.choice(len(present_sites), size=num_spins, replace=False))

    return [present_sites[i] for i in idx]
