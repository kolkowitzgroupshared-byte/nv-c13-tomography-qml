from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from .fine_decay import fine_decay, _comb_quartic_powerlaw


def _synthesize_comb_only(taus_sec, p):
    return fine_decay(
        tau_us=taus_sec * 1e6,
        baseline=float(p.get("baseline", 1.0)),
        comb_contrast=float(p.get("comb_contrast", 0.6)),
        revival_time=float(p.get("revival_time", 37.0)),
        width0_us=float(p.get("width0_us", 6.0)),
        T2_ms=float(p.get("T2_ms", 0.08)),
        T2_exp=float(p.get("T2_exp", 1.0)),
        amp_taper_alpha=float(p.get("amp_taper_alpha", 0.0)),
        width_slope=float(p.get("width_slope", 0.0)),
        revival_chirp=float(p.get("revival_chirp", 0.0)),
        osc_amp=0.0,  # <- explicitly no oscillation when synthesizing comb-only
        osc_f0=0.0,
        osc_f1=0.0,
        osc_phi0=0.0,
        osc_phi1=0.0,
    )


def add_charge_pedestal(
    y_core,
    taus_us,
    gate_G,
    *,
    A_ch=0.03,  # amplitude of the pedestal (0..~0.1 reasonable)
    T_ch_us=None,  # optional readout-specific decay; None→use no extra decay
    baseline=0.6,
):
    """
    y_core: baseline - depth(τ)*E(τ)   (your existing physical echo)
    gate_G: 0 at τ≈0, peaks at k*Trev  (same comb you already compute, normalized 0..1)
    A_ch  : pedestal amplitude (fraction of total scale)
    T_ch_us: if set, pedestal has its own envelope exp[-(τ/T_ch)^1] (often > T2_us)
    """
    tau = np.asarray(taus_us, float)
    G = np.clip(np.asarray(gate_G, float), 0.0, 1.0)

    # Optional slow decay for charge gain (often longer than spin T2)
    if T_ch_us is not None and T_ch_us > 0:
        E_ch = np.exp(-(tau / float(T_ch_us)))
    else:
        E_ch = 1.0

    P = A_ch * G * E_ch  # strictly >= 0

    # Headroom safety: don’t exceed physical maximum (≈1.0), but do it smoothly
    headroom = 1.0 - y_core
    P = np.minimum(P, np.maximum(0.0, headroom))

    return y_core + P


def apply_readout_gain(y_core, gate_G, *, beta=0.08, baseline=0.6):
    """
    y_core is already on your 0..1-ish PL scale with baseline ~0.6.
    We remap around baseline so that a gain >1 lifts toward/above baseline.
    """
    G = np.clip(np.asarray(gate_G, float), 0.0, 1.0)

    # deviation from baseline, then apply a gain that also adds a small offset piece:
    #   y_raw = baseline + (y_core - baseline) * (1 + beta*G) + beta*G*(1 - baseline)
    # The last term acts like a gain-induced offset in counts.
    y_out = (
        baseline + (y_core - baseline) * (1.0 + beta * G) + beta * G * (1.0 - baseline)
    )

    # Smooth headroom cap (no hard clip)
    return np.minimum(y_out, 1.0 - 1e-6)


def revivals_only_mapping(microscopic, taus_s, p):
    """
    Gate microscopic deviations to revivals AND add a zero-mean oscillatory term
    so the signal can go above baseline near revivals (as seen experimentally).

    p expects (in addition to your usual fine params):
      baseline, comb_contrast,
      revival_time (us), width0_us (us), T2_ms, T2_exp,
      amp_taper_alpha, width_slope, revival_chirp,
    """
    # ---- unpack ----
    baseline = float(p.get("baseline", 0.6))
    comb_contrast = float(p.get("comb_contrast", 0.4))
    Trev_us = max(1e-9, float(p.get("revival_time", 37.3)))
    w0_us = max(1e-9, float(p.get("width0_us", 6.0)))
    T2_ms = float(p.get("T2_ms", 0.08))
    T2_exp = float(p.get("T2_exp", 1.2))
    taper = float(p.get("amp_taper_alpha", 0.0))
    w_slope = float(p.get("width_slope", 0.0))
    chirp = float(p.get("revival_chirp", 0.0))
    # amplitude around 0d 0
    taus_us = np.asarray(taus_s, float) * 1e6
    # ---- comb mask (0..1), tightened by 'power' ----
    tau_max = float(np.nanmax(taus_us)) if taus_us.size else 0.0
    n_guess = max(1, min(64, int(np.ceil(1.2 * tau_max / Trev_us)) + 1))
    mask = _comb_quartic_powerlaw(
        taus_us, Trev_us, w0_us, taper, w_slope, chirp, n_guess
    )
    # microscopic factor m(τ) with m(0)=1
    m = np.asarray(microscopic, float)
    taus_us = np.asarray(taus_s, float) * 1e6  # x-axis in μs
    m = np.asarray(microscopic, float)

    # envelope
    T2_us = max(1e-9, 1000.0 * T2_ms)
    E = np.exp(-((taus_us / T2_us) ** T2_exp))
    # --- revival gate (≈0 at τ≈0, peaks at k*Trev) ---
    y_core = baseline - comb_contrast * m * mask * E
    # --- Physically motivated gains at revivals ---
    # A) charge pedestal
    # y_out = add_charge_pedestal(
    #     y_core, taus_us, mask, A_ch=0.03, T_ch_us=300.0, baseline=baseline
    # )
    # B) multiplicative gain
    # y_out = apply_readout_gain(y_core, mask, beta=0.08, baseline=baseline)
    return y_core
