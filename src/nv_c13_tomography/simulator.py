from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from .constants import GAMMA_C13_HZ_PER_T, GAUSS_TO_TESLA
from .fine_decay import fine_decay
from .catalog import CatalogSite


@dataclass
class SimResult:
    taus_us: np.ndarray
    echo: np.ndarray
    aux: Dict


def Mk_from_catalog(site: CatalogSite, tau_s: np.ndarray) -> np.ndarray:
    # 1 - 2 kappa sin^2(pi f0 tau) sin^2(pi f_m1 tau)
    return 1.0 - 2.0 * site.kappa * (np.sin(np.pi * site.f0_Hz * tau_s) ** 2) * (
        np.sin(np.pi * site.f_m1_Hz * tau_s) ** 2
    )


def compute_echo_from_catalog(sites: List[CatalogSite], tau_s: np.ndarray) -> np.ndarray:
    y = np.ones_like(tau_s, dtype=float)
    for s in sites:
        y *= Mk_from_catalog(s, tau_s)
    return y


def choose_sites(
    rng: np.random.Generator,
    present_sites: List[CatalogSite],
    num_spins: Optional[int],
    selection_mode: str,
) -> List[CatalogSite]:
    if num_spins is None or num_spins >= len(present_sites):
        return present_sites

    if selection_mode == "top_kappa":
        order = np.argsort([-abs(s.kappa) for s in present_sites])
        idx = order[:num_spins]
        return [present_sites[i] for i in idx]

    # fallback uniform
    idx = np.sort(rng.choice(len(present_sites), size=num_spins, replace=False))
    return [present_sites[i] for i in idx]


def simulate_catalog_mode(
    *,
    catalog: List[CatalogSite],
    tau_range_us: Tuple[float, float],
    n_tau: int,
    B_vec_G: Tuple[float, float, float],
    abundance_fraction: float,
    num_spins: Optional[int],
    selection_mode: str,
    rng_seed: int,
    fine_params: Optional[Dict] = None,
    nv_orientation: Optional[Tuple[int, int, int]] = None,
    distance_cutoff_A: Optional[float] = None,
    f_band_kHz: Optional[Tuple[float, float]] = None,
    reuse_present_mask: bool = True,
) -> SimResult:
    rng = np.random.default_rng(int(rng_seed))

    taus_us = np.linspace(float(tau_range_us[0]), float(tau_range_us[1]), n_tau)
    tau_s = taus_us * 1e-6

    # filter catalog
    sites = catalog
    if nv_orientation is not None:
        sites = [s for s in sites if s.orientation == tuple(nv_orientation)]
    if distance_cutoff_A is not None:
        sites = [s for s in sites if (s.distance_A is not None and s.distance_A < float(distance_cutoff_A))]
    if f_band_kHz is not None:
        lo, hi = map(float, f_band_kHz)
        def in_band(x_hz: float) -> bool:
            x = x_hz / 1e3
            return (x >= lo) and (x <= hi)
        sites = [s for s in sites if in_band(s.f_minus_Hz) and in_band(s.f_plus_Hz)]

    if len(sites) == 0:
        echo = np.ones_like(taus_us)
        return SimResult(taus_us=taus_us, echo=echo, aux={"N_candidates": 0, "chosen": []})

    # quenched occupancy
    if reuse_present_mask:
        present_mask = rng.random(len(sites)) < float(abundance_fraction)
    else:
        present_mask = rng.random(len(sites)) < float(abundance_fraction)

    present = [s for s, m in zip(sites, present_mask) if m]
    chosen = choose_sites(rng, present, num_spins, selection_mode) if len(present) else []

    echo = compute_echo_from_catalog(chosen, tau_s) if len(chosen) else np.ones_like(tau_s)

    # apply phenomenology (optional)
    if fine_params is not None:
        echo = fine_decay(
            taus_us,
            baseline=float(fine_params.get("baseline", 0.6)),
            comb_contrast=float(fine_params.get("comb_contrast", 0.4)),
            revival_time=float(fine_params.get("revival_time_us", 37.0)),
            width0_us=float(fine_params.get("width0_us", 6.0)),
            T2_ms=float(fine_params.get("T2_ms", 0.08)),
            T2_exp=float(fine_params.get("T2_exp", 1.2)),
            amp_taper_alpha=float(fine_params.get("amp_taper_alpha", 0.0)),
            width_slope=float(fine_params.get("width_slope", 0.0)),
            revival_chirp=float(fine_params.get("revival_chirp", 0.0)),
            osc_amp=float(fine_params.get("osc_amp", 0.0)),
            osc_f0=float(fine_params.get("osc_f0", 0.0)),
            osc_phi0=float(fine_params.get("osc_phi0", 0.0)),
            osc_f1=float(fine_params.get("osc_f1", 0.0)),
            osc_phi1=float(fine_params.get("osc_phi1", 0.0)),
        )

    aux = {
        "N_candidates": len(sites),
        "N_present": len(present),
        "N_chosen": len(chosen),
        "chosen_ids": [s.site_id for s in chosen],
    }
    return SimResult(taus_us=taus_us, echo=echo, aux=aux)
