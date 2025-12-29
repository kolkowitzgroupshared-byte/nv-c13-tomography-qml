from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json

from .rng import spawn_streams, choose_sites
from .hyperfine import make_R_NV, compute_hyperfine_components
from .echo_core import compute_echo_signal, compute_echo_signal_from_catalog
from .fine_decay import fine_decay, _comb_quartic_powerlaw  # if needed for revivals_only_mapping
from .echo_mapping import revivals_only_mapping, _synthesize_comb_only# plus your IO helper read_hyperfine_table_safe (keep here or put in io.py)

def read_hyperfine_table_safe(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    # find first data row that starts with an integer (skip headers/junk)
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    def _is_int_start(s: str) -> bool:
        s = s.lstrip()
        if not s:
            return False
        t = s.split()[0]
        try:
            int(t)
            return True
        except Exception:
            return False

    try:
        data_start = next(i for i, line in enumerate(lines) if _is_int_start(line))
    except StopIteration:
        raise RuntimeError(f"Could not locate data start in hyperfine file: {path}")

    HF_COLS = [
        "index",
        "distance",
        "x",
        "y",
        "z",
        "Axx",
        "Ayy",
        "Azz",
        "Axy",
        "Axz",
        "Ayz",
    ]
    # primary path: pandas
    try:
        df = pd.read_csv(
            path,
            sep=r"\s+",  # robust whitespace split
            engine="python",
            comment="#",  # ignore commented tails
            header=None,
            names=HF_COLS,
            usecols=list(range(11)),  # ensure exactly 11 cols
            skiprows=data_start,
            na_filter=False,
        )
        # enforce dtypes
        df = df.astype(
            {
                "index": int,
                "distance": float,
                "x": float,
                "y": float,
                "z": float,
                "Axx": float,
                "Ayy": float,
                "Azz": float,
                "Axy": float,
                "Axz": float,
                "Ayz": float,
            },
            errors="ignore",
        )
        return df
    except Exception as e:
        # fallback: numpy → DataFrame
        arr = np.loadtxt(
            path,
            comments="#",
            dtype=float,
            ndmin=2,
        )
        if arr.shape[1] < 11:
            raise RuntimeError(
                f"Expected ≥11 columns, found {arr.shape[1]} in {path}"
            ) from e
        arr = arr[:, :11]
        df = pd.DataFrame(arr, columns=HF_COLS)
        # index is float now; coerce to int safely
        df["index"] = df["index"].round().astype(int)
        return df


# ---------- 0) IO ----------
def load_catalog(path_json: str) -> List[Dict]:
    """Load ESEEM catalog (with κ and f± in Hz) written by your builder."""
    with open(path_json, "r") as f:
        return json.load(f)


def _ori_tuple(o) -> Tuple[int, int, int]:
    return tuple(int(x) for x in o)


# =============================================================================
# Main simulator
# =============================================================================
def simulate_random_spin_echo(
    hyperfine_path,
    tau_range_us,
    num_spins=30,
    num_realizations=1,
    B_vec_T = np.array([0.0, 0.0, 50e-4]),  # Tesla
    distance_cutoff=None,
    Ak_min_kHz=None,  # keep if A∥ ≥ Ak_min_kHz (if set)
    Ak_max_kHz=None,  # keep if A∥ ≤ Ak_max_kHz (if set)
    Ak_abs=True,  # compare |A∥| if True, signed A∥ if False
    R_NV=np.eye(3),
    fine_params=None,  # set None for microscopic-only
    abundance_fraction=0.011,
    rng_seed=None,
    run_salt=None,
    randomize_positions=False,  # keep False for single NV
    selection_mode="top_Apar",
    ensure_unique_across_realizations=False,  # usually False for fixed NV
    annotate_from_realization=0,
    keep_nv_orientation=True,  # keep True for single NV
    fixed_site_ids=None,  # exact sites to include
    fixed_presence_mask=None,  # boolean mask of length N_sites
    reuse_present_mask=True,  # draw Bernoulli once and reuse (quenched)
    catalog_json_path=None,  # if not None, use catalog instead of hyperfine_path
    nv_orientation=None,  # e.g. (1,1,1); if None, keep all orientations
):
    """
    Returns:
      taus_us, avg_signal, aux
      aux = {
        "positions": (N,3) of annotated realization (NV frame),
        "site_info": [{site_id, Apar_kHz, r}, ...],
        "revivals_us": array of k*revival_time for plotting,
        "picked_ids_per_realization": [[...], ...],
        "stats": {...}
      }
    """
    # Time axis
    taus_s = np.linspace(float(tau_range_us[0]), float(tau_range_us[1]), num=600) * 1e-6

    # RNG
    rng_streams = spawn_streams(rng_seed, max(1, num_realizations), run_salt=run_salt)

    # Rotate B into NV frame once
    # Rotate B into NV frame once (still used by the hyperfine path)
    B_vec_NV = R_NV @ B_vec_T
    B_hat_NV = B_vec_NV / np.linalg.norm(B_vec_NV)

    # -------------------------------------------------------------------------
    # A) Use raw hyperfine table (original behavior)
    # -------------------------------------------------------------------------
    if catalog_json_path is None:
        df = read_hyperfine_table_safe(hyperfine_path)
        if distance_cutoff is not None:
            df = df[df["distance"] < distance_cutoff]

        sites = []
        for _, row in df.iterrows():
            A = (
                np.array(
                    [
                        [row.Axx, row.Axy, row.Axz],
                        [row.Axy, row.Ayy, row.Ayz],
                        [row.Axz, row.Ayz, row.Azz],
                    ],
                    float,
                )
                * 1e6
            )  # MHz -> Hz
            A_nv = R_NV @ A @ R_NV.T

            # Apparent A_parallel for current B (NV frame)
            A_par, _ = compute_hyperfine_components(A_nv, B_hat_NV)

            # A∥ filter (existing logic)
            A_par_kHz = (abs(A_par) if Ak_abs else A_par) / 1e3
            keep_A = True
            if Ak_min_kHz is not None:
                keep_A &= A_par_kHz >= float(Ak_min_kHz)
            if Ak_max_kHz is not None:
                keep_A &= A_par_kHz <= float(Ak_max_kHz)
            if not keep_A:
                continue

            pos_crystal = np.array([row.x, row.y, row.z], float)
            pos_nv = R_NV @ pos_crystal
            sites.append(
                {
                    "site_id": int(row["index"]),
                    "A0": A_nv,
                    "pos0": pos_nv,
                    "dist": float(row.distance),
                    "Apar_Hz": float(A_par),
                }
            )
    # -------------------------------------------------------------------------
    # B) Use pre-built ESEEM catalog (kappa, f± in lab frame)
    # -------------------------------------------------------------------------
    else:
        # Use pre-built ESEEM catalog (JSON list of dicts)
        catalog = load_catalog(catalog_json_path)

        # Optional: filter on NV orientation, e.g. (1,1,1)
        if nv_orientation is not None:
            ori_tgt = _ori_tuple(nv_orientation)
            catalog = [
                rec for rec in catalog if _ori_tuple(rec["orientation"]) == ori_tgt
            ]
            print(f"After orientation filter {ori_tgt}: {len(catalog)} records")

        # Optional: distance cutoff (Å)
        if distance_cutoff is not None:
            dmax = float(distance_cutoff)
            catalog = [rec for rec in catalog if rec.get("distance_A", np.inf) < dmax]
            print(f"After distance cutoff < {dmax} Å: {len(catalog)} records")

        f_minus_all = np.array([rec["f_minus_Hz"] for rec in catalog]) / 1e3
        f_plus_all = np.array([rec["f_plus_Hz"] for rec in catalog]) / 1e3
        print(f"f_- range (kHz): {f_minus_all.min():.1f} – {f_minus_all.max():.1f}")
        print(f"f_+ range (kHz): {f_plus_all.min():.1f} – {f_plus_all.max():.1f}")

        sites = []
        for rec in catalog:
            # -------- frequency-band filter using f_- and f_+ (in kHz) --------
            f_minus_Hz = float(rec["f_minus_Hz"])
            f_plus_Hz = float(rec["f_plus_Hz"])
            f_minus_kHz = f_minus_Hz / 1e3
            f_plus_kHz = f_plus_Hz / 1e3

            keep_f = True
            if (Ak_min_kHz is not None) or (Ak_max_kHz is not None):
                # Require *both* f_- and f_+ to lie inside [Ak_min_kHz, Ak_max_kHz]
                in_minus = True
                in_plus = True

                if Ak_min_kHz is not None:
                    lo = float(Ak_min_kHz)
                    in_minus &= f_minus_kHz >= lo
                    in_plus &= f_plus_kHz >= lo

                if Ak_max_kHz is not None:
                    hi = float(Ak_max_kHz)
                    in_minus &= f_minus_kHz <= hi
                    in_plus &= f_plus_kHz <= hi

                # keep only if BOTH lines are inside the band
                keep_f = in_minus and in_plus

            if not keep_f:
                continue

            # ------------------------------------------------------------------
            pos = np.array(
                [
                    rec.get("x_A", rec.get("x", np.nan)),
                    rec.get("y_A", rec.get("y", np.nan)),
                    rec.get("z_A", rec.get("z", np.nan)),
                ],
                float,
            )

            sites.append(
                {
                    "site_id": int(rec["site_index"]),
                    # keep keys expected by the rest of the code:
                    "A0": None,  # not used in catalog mode
                    "pos0": pos,
                    "dist": float(rec.get("distance_A", np.nan)),
                    # catalog-specific fields for time-domain M(τ)
                    "kappa": float(rec["kappa"]),
                    "f0_Hz": float(rec["fI_Hz"]),  # f_0  (ms = 0 manifold)
                    "f_m1_Hz": float(rec["omega_ms_Hz"]),  # f_-1 (ms = -1 manifold)
                    # sum/diff lines and weights
                    "f_minus_Hz": f_minus_Hz,
                    "f_plus_Hz": f_plus_Hz,
                    "line_w_minus": float(rec.get("line_w_minus", 0.0)),
                    "line_w_plus": float(rec.get("line_w_plus", 0.0)),
                }
            )

    N_candidates = len(sites)
    print(
        f"Total candidate sites after catalog filters "
        f"(orientation, distance, f_-/f_+ band): {N_candidates}"
    )
    if N_candidates == 0:
        taus_us = taus_s * 1e6
        flat = np.ones_like(taus_us)
        return (
            taus_us,
            flat,
            {
                "positions": None,
                "site_info": [],
                "revivals_us": None,
                "picked_ids_per_realization": [],
                "stats": {},
            },
        )

    id_to_idx = {s["site_id"]: i for i, s in enumerate(sites)}

    present_mask_global = None
    if (
        (fixed_site_ids is None)
        and (fixed_presence_mask is None)
        and reuse_present_mask
    ):
        rng_once = rng_streams[0]
        present_mask_global = rng_once.random(N_candidates) < abundance_fraction

    # Containers
    all_signals = []
    picked_ids_per_realization = []
    present_counts = []
    chosen_counts = []
    anno_positions = None
    anno_site_info = None
    anno_rev_times = None
    used_site_ids = set()

    for r in range(num_realizations):
        rng_r = rng_streams[r]

        # Decide occupancy
        if fixed_site_ids:
            present_idxs = np.array(
                [id_to_idx[i] for i in fixed_site_ids if i in id_to_idx], int
            )
            present_mask = np.zeros(N_candidates, dtype=bool)
            present_mask[present_idxs] = True
        elif fixed_presence_mask is not None:
            mask = np.asarray(fixed_presence_mask, bool)
            if mask.size != N_candidates:
                raise ValueError(
                    "fixed_presence_mask length does not match candidate site count."
                )
            present_mask = mask
            present_idxs = np.flatnonzero(present_mask)
        elif present_mask_global is not None:
            present_mask = present_mask_global
            present_idxs = np.flatnonzero(present_mask)
        else:
            present_mask = rng_r.random(N_candidates) < abundance_fraction
            present_idxs = np.flatnonzero(present_mask)

        present_counts.append(int(present_mask.sum()))
        if present_idxs.size == 0:
            all_signals.append(np.ones_like(taus_s))
            picked_ids_per_realization.append([])
            continue

        present_sites = [sites[i] for i in present_idxs]

        # Optional cross-realization uniqueness (usually False for fixed NV)
        if ensure_unique_across_realizations:
            filtered = [s for s in present_sites if s["site_id"] not in used_site_ids]
            if len(filtered) >= max(1, num_spins if num_spins is not None else 1):
                present_sites = filtered

        # Choose final subset (or take all)
        chosen_sites = choose_sites(rng_r, present_sites, num_spins, selection_mode)
        chosen_counts.append(len(chosen_sites))
        picked_ids = [s["site_id"] for s in chosen_sites]
        picked_ids_per_realization.append(picked_ids)
        used_site_ids.update(picked_ids)

        # Compute echo: either from raw hyperfine tensors or from catalog
        if catalog_json_path is None:
            tensors = [s["A0"] for s in chosen_sites]
            signal = compute_echo_signal(tensors, taus_s, B_vec_NV)
        else:
            signal = compute_echo_signal_from_catalog(chosen_sites, taus_s)

        all_signals.append(signal)

        # Annotation (first realization by default)
        if r == annotate_from_realization:
            anno_positions = (
                np.array([s["pos0"] for s in chosen_sites]) if chosen_sites else None
            )

            if catalog_json_path is None:
                anno_site_info = [
                    {
                        "site_id": s["site_id"],
                        "Apar_kHz": float(
                            abs(compute_hyperfine_components(s["A0"], B_hat_NV)[0])
                            / 1e3
                        ),
                        "r": float(np.linalg.norm(s["pos0"])),
                    }
                    for s in chosen_sites
                ]
            else:
                # use catalog data directly (A∥ + f_- / f_+)
                anno_site_info = [
                    {
                        "site_id": s["site_id"],
                        "Apar_kHz": float(abs(s.get("Apar_Hz", 0.0)) / 1e3),
                        "r": float(np.linalg.norm(s["pos0"])),
                        "f_minus_kHz": float(s.get("f_minus_Hz", 0.0) / 1e3),
                        "f_plus_kHz": float(s.get("f_plus_Hz", 0.0) / 1e3),
                    }
                    for s in chosen_sites
                ]

            if fine_params is not None and "revival_time" in fine_params:
                revT_us = float(fine_params["revival_time"])
                kmax = int(np.ceil((taus_s.max() * 1e6) / revT_us))
                anno_rev_times = np.arange(0, kmax + 1) * revT_us

    # Average (for single realization this is just identity)
    avg_signal = np.mean(all_signals, axis=0)

    # phenomenological gating
    if fine_params is not None:
        if np.nanmax(avg_signal) - np.nanmin(avg_signal) < 1e-4:
            # no microscopic modulation -> synthesize clean comb at your baseline
            avg_signal = _synthesize_comb_only(taus_s, fine_params)
        else:
            # <-- key line: gate deviations so oscillations live only near revivals
            avg_signal = revivals_only_mapping(avg_signal, taus_s, fine_params)

    stats = {
        "N_candidates": N_candidates,
        "abundance_fraction": float(abundance_fraction),
        "present_counts": present_counts,
        "chosen_counts": chosen_counts,
    }

    return (
        taus_s * 1e6,
        avg_signal,
        {
            "positions": anno_positions,
            "site_info": anno_site_info if anno_site_info is not None else [],
            "revivals_us": anno_rev_times,
            "picked_ids_per_realization": picked_ids_per_realization,
            "stats": stats,
            "all_candidate_positions": np.array(
                [s["pos0"] for s in sites], float
            ),  # NEW
        },
    )
