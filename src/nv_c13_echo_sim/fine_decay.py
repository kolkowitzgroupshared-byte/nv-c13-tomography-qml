from __future__ import annotations
import numpy as np

try:
    from numba import njit
except Exception:
    def njit(*_a, **_k):
        def wrap(fn): return fn
        return wrap

def fine_decay(
    tau_us,
    baseline=1.0,
    comb_contrast=0.6,
    revival_time=37.0,
    width0_us=6.0,
    T2_ms=0.08,
    T2_exp=1.0,
    amp_taper_alpha=0.0,
    width_slope=0.0,
    revival_chirp=0.0,
    # oscillation knobs (use THESE names everywhere)
    osc_amp=0.0,
    osc_f0=0.0,      # cycles/us (MHz)
    osc_phi0=0.0,
    osc_f1=0.0,
    osc_phi1=0.0,
):
    tau = np.asarray(tau_us, dtype=float).ravel()

    width0_us = max(1e-9, float(width0_us))
    revival_time = max(1e-9, float(revival_time))
    T2_us = max(1e-9, 1000.0 * float(T2_ms))
    T2_exp = float(T2_exp)

    envelope = np.exp(-((tau / T2_us) ** T2_exp))

    tau_max = float(np.nanmax(tau)) if tau.size else 0.0
    n_guess = max(1, min(64, int(np.ceil(1.2 * tau_max / revival_time)) + 1))

    comb = _comb_quartic_powerlaw(
        tau, revival_time, width0_us, amp_taper_alpha, width_slope, revival_chirp, n_guess
    )

    if (osc_amp != 0.0) and (osc_f0 != 0.0 or osc_f1 != 0.0):
        s0 = np.sin(np.pi * osc_f0 * tau + osc_phi0)
        s1 = np.sin(np.pi * osc_f1 * tau + osc_phi1)
        beat = (s0 * s0) * (s1 * s1)
        mod = comb_contrast - osc_amp * beat
    else:
        mod = comb_contrast

    return baseline - envelope * mod * comb


@njit
def _comb_quartic_powerlaw(tau, revival_time, width0_us, amp_taper_alpha, width_slope, revival_chirp, n_guess):
    n = tau.shape[0]
    out = np.zeros(n, dtype=np.float64)

    tmax = 0.0
    for i in range(n):
        if tau[i] > tmax:
            tmax = tau[i]

    for k in range(n_guess):
        mu_k = k * revival_time * (1.0 + k * revival_chirp)
        w_k = width0_us * (1.0 + k * width_slope)
        if w_k <= 0.0:
            continue
        if mu_k > tmax + 5.0 * w_k:
            break

        amp_k = 1.0 / ((1.0 + k) ** amp_taper_alpha)
        inv_w4 = 1.0 / (w_k**4)

        for i in range(n):
            x = tau[i] - mu_k
            out[i] += amp_k * np.exp(-(x * x) * (x * x) * inv_w4)

    return out
