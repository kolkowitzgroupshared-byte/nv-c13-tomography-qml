# noise.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class NoiseSpec:
    # photon/shot model (approximate binomial with variance ~ p(1-p)/N)
    use_shot: bool = True
    photons_mean: float = 3_000.0  # mean detected photons per point at y=1

    # additive electronics noise
    add_gauss_std: float = 0.003  # ~0.3% RMS

    # slow drifts per trace
    mult_gain_std: float = 0.02  # multiplicative gain ~ N(1,σ)
    base_offset_std: float = 0.01  # additive offset ~ N(0,σ)

    # correlated noise across tau
    use_1f: bool = True
    onef_strength: float = 0.004  # overall RMS of 1/f component
    onef_alpha: float = 1.0  # spectral slope: 0=white, 1=pink

    use_ar1: bool = False  # alternative to 1/f if you prefer
    ar1_rho: float = 0.98  # correlation
    ar1_sigma: float = 0.002  # innovations std

    # timing jitter (distorts x-axis slightly)
    timing_jitter_std_us: float = 0.01  # std of τ jitter (μs), applied pointwise

    # small per-trace frequency detuning (mimics MW/B-field drift)
    freq_detune_ppm: float = 200.0  # ppm of MHz-scale features; set 0 to disable

    # rare spikes/outliers
    glitch_prob: float = 0.001
    glitch_magnitude: float = 0.1  # absolute kick

    # quantization
    quantize_bits: int = 0  # 0 to disable; else 12, 14, 16, ...

    # bounds
    clip_lo: float = -0.1  # allow slight under/overshoot pre-clip
    clip_hi: float = 1.2


def _gen_1f_noise(T: int, rng: np.random.Generator, alpha=1.0, rms=0.005):
    """Synthesize length-T 1/f^alpha noise with target RMS."""
    # frequency grid
    freqs = np.fft.rfftfreq(T)
    amp = np.ones_like(freqs)
    # avoid division by zero at DC; leave DC random but small
    amp[1:] = 1.0 / (freqs[1:] ** (alpha / 2.0))  # /2 because we’ll mirror energy
    # complex spectrum with random phases
    phases = rng.uniform(0, 2 * np.pi, size=freqs.shape)
    spec = amp * (np.cos(phases) + 1j * np.sin(phases))
    # inverse FFT to time domain
    x = np.fft.irfft(spec, n=T)
    # normalize to unit RMS then scale
    x = x / (np.std(x) + 1e-12)
    return rms * x


def _gen_ar1_noise(T: int, rng: np.random.Generator, rho=0.98, sigma=0.002):
    e = rng.standard_normal(T) * sigma
    x = np.empty(T, float)
    x[0] = e[0] / max(1e-6, (1 - rho))
    for t in range(1, T):
        x[t] = rho * x[t - 1] + e[t]
    return x


def apply_noise_pipeline(
    y_clean: np.ndarray, taus_us: np.ndarray, rng: np.random.Generator, spec: NoiseSpec
) -> np.ndarray:
    y = y_clean.astype(float, copy=True)
    T = y.shape[0]

    # (0) small per-trace detuning → stretch tau slightly
    if spec.freq_detune_ppm and np.any(np.diff(taus_us) > 0):
        scale = 1.0 + 1e-6 * spec.freq_detune_ppm * rng.standard_normal()
        # resample y at tau' = tau * scale
        tau_scaled = taus_us * scale
        y = np.interp(taus_us, tau_scaled, y, left=y[0], right=y[-1])

    # (1) multiplicative gain & baseline offset (per trace)
    if spec.mult_gain_std > 0:
        gain = 1.0 + spec.mult_gain_std * rng.standard_normal()
        y *= gain
    if spec.base_offset_std > 0:
        y += spec.base_offset_std * rng.standard_normal()

    # (2) correlated noise across τ
    if spec.use_1f and spec.onef_strength > 0:
        y += _gen_1f_noise(T, rng, alpha=spec.onef_alpha, rms=spec.onef_strength)
    elif spec.use_ar1:
        y += _gen_ar1_noise(T, rng, rho=spec.ar1_rho, sigma=spec.ar1_sigma)

    # (3) timing jitter: perturb τ and resample (pointwise)
    if spec.timing_jitter_std_us > 0:
        dt = rng.standard_normal(T) * spec.timing_jitter_std_us
        tau_jit = np.clip(taus_us + dt, taus_us.min(), taus_us.max())
        tau_jit.sort()  # preserve monotonicity; yields slight local stretching
        y = np.interp(taus_us, tau_jit, y)

    # (4) shot/readout noise (approximate binomial/Poisson)
    if spec.use_shot and spec.photons_mean > 0:
        # approximate: counts ~ Poisson(mu = photons_mean * y_clipped)
        lam = np.clip(y, 0.0, 1.0) * spec.photons_mean
        counts = rng.poisson(lam)
        y = counts / max(1.0, spec.photons_mean)

    # (5) additive Gaussian electronics noise
    if spec.add_gauss_std > 0:
        y += spec.add_gauss_std * rng.standard_normal(T)

    # (6) rare glitches/outliers
    if spec.glitch_prob > 0 and spec.glitch_magnitude > 0:
        mask = rng.random(T) < spec.glitch_prob
        y[mask] += spec.glitch_magnitude * rng.standard_normal(mask.sum())

    # (7) quantization
    if spec.quantize_bits and spec.quantize_bits > 0:
        levels = 2**spec.quantize_bits
        lo, hi = 0.0, 1.0  # assume normalized signal; adjust if needed
        step = (hi - lo) / (levels - 1)
        y = np.round(np.clip(y, lo, hi) / step) * step

    # final clip
    return np.clip(y, spec.clip_lo, spec.clip_hi)

