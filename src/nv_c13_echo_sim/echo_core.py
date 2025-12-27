from __future__ import annotations
import numpy as np
from typing import List, Dict
from .hyperfine import compute_hyperfine_components, Mk_tau
from .constants import gamma_C13


# =============================================================================
# Single-nucleus echo factor Mk(τ) and many-spin product
# =============================================================================
def Mk_tau(A_par_Hz, A_perp_Hz, omegaI_Hz, tau_s):
    """
    Exact single-nucleus ESEEM factor:
      ω_{-1} = sqrt((ωI - A_par)^2 + A_perp^2)
      ω_0    = ωI
      ω±     = ω_{-1} ± ω_0
      κ      = (A_perp^2) / (ω_{-1}^2)
      M(τ)   = 1 - κ * sin^2(π ω_+ τ) * sin^2(π ω_- τ)
    All freqs in Hz; τ in seconds.
    """
    wI = omegaI_Hz
    w_m1 = np.sqrt((wI - A_par_Hz) ** 2 + A_perp_Hz**2)
    w0 = wI
    w_plus = w_m1 + w0
    w_minus = w_m1 - w0
    kappa = (A_perp_Hz**2) / (w_m1**2 + 1e-30)

    # sin arguments need cycles → use π*freq*τ because sin^2(ω τ /2) with ω=2π f ⇒ sin^2(π f τ)
    return 1.0 - 2 * kappa * (np.sin(np.pi * w_plus * tau_s) ** 2) * (
        np.sin(np.pi * w_minus * tau_s) ** 2
    )


def compute_echo_signal(
    hyperfine_tensors,
    tau_array_s,
    B_field_vec_T,
    sigma_B_G=0.0,
    rng=None,
):
    B_vec = np.array(B_field_vec_T, float)
    if sigma_B_G > 0.0 and rng is not None:
        B_vec = B_vec + 1e-4 * rng.normal(0.0, sigma_B_G, size=3)  # G→T

    B_mag = np.linalg.norm(B_vec)
    if B_mag == 0.0:
        raise ValueError("B-field magnitude is zero.")
    B_unit = B_vec / B_mag
    omega_L = gamma_C13 * B_mag  # Hz

    signal = np.empty_like(tau_array_s, dtype=float)

    for i, tau in enumerate(tau_array_s):
        Mk_prod = 1.0
        for A_tensor in hyperfine_tensors:
            A_par, A_perp = compute_hyperfine_components(A_tensor, B_unit)
            Mk = Mk_tau(A_par, A_perp, omega_L, tau)
            Mk_prod *= Mk
        signal[i] = Mk_prod
    return signal


def Mk_from_catalog_rec(rec, tau_array_s):
    """
    ESEEM factor using (kappa, f0, f-1) from the catalog.

    rec must have:
      - 'kappa'
      - 'fI_Hz'       ≡ f0
      - 'omega_ms_Hz' ≡ f-1
    All freqs in Hz, tau_array_s in seconds.
    """

    kappa = float(rec["kappa"])
    f0 = float(rec["f0_Hz"])  # from fI_Hz
    f_m1 = float(rec["f_m1_Hz"])  # from omega_ms_Hz
    tau = np.asarray(tau_array_s, float)

    return 1.0 - 2.0 * kappa * (np.sin(np.pi * f0 * tau) ** 2) * (
        np.sin(np.pi * f_m1 * tau) ** 2
    )


def compute_echo_signal_from_catalog(chosen_sites, tau_array_s):
    signal = np.ones_like(tau_array_s, dtype=float)
    for rec in chosen_sites:
        signal *= Mk_from_catalog_rec(rec, tau_array_s)
    return signal

