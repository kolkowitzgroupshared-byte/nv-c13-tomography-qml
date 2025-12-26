from __future__ import annotations
import numpy as np
from typing import Tuple
from .constants import gamma_C13

def _orthonormal_basis_from_z(z):
    """Build an ONB {ez=z/|z|, e1, e2}."""
    ez = np.asarray(z, float)
    ez /= np.linalg.norm(ez)
    tmp = np.array([1.0, 0.0, 0.0]) if abs(ez[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = tmp - ez * np.dot(tmp, ez)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(ez, e1)
    return ez, e1, e2


def make_R_NV(nv_axis_crystal):
    """
    nv_axis_crystal: tuple/list/np.array like (±1, ±1, ±1)
    Returns R such that v_NV = R @ v_crystal and ez_NV aligns with nv_axis.
    """
    n = np.asarray(nv_axis_crystal, float)
    n /= np.linalg.norm(n)  # unit
    # reuse your ONB constructor but for the NV axis now
    ez, e1, e2 = _orthonormal_basis_from_z(n)  # ez=n, e1,e2 ⟂ n
    # rows are basis vectors expressed in crystal coords → left-multiply for coords in NV basis
    return np.vstack([e1, e2, ez])


def compute_hyperfine_components(A_tensor, dir_hat):
    """
    Return (A_par, A_perp) in Hz, defined w.r.t. the *direction* dir_hat
    (usually the magnetic-field unit vector, not the NV axis).
    """
    ez = np.asarray(dir_hat, float)
    ez /= np.linalg.norm(ez)
    tmp = np.array([1.0, 0.0, 0.0]) if abs(ez[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = tmp - ez * np.dot(tmp, ez)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(ez, e1)
    A_par = ez @ A_tensor @ ez
    A_perp = np.sqrt((e1 @ A_tensor @ ez) ** 2 + (e2 @ A_tensor @ ez) ** 2)
    return A_par, A_perp


def U_111_to_cubic():
    ex = np.array([1.0, -1.0, 0.0])
    ex /= np.linalg.norm(ex)  # [1,-1,0]/√2
    ez = np.array([1.0, 1.0, 1.0])
    ez /= np.linalg.norm(ez)  # [1, 1,1]/√3
    ey = np.cross(ez, ex)
    ey /= np.linalg.norm(ey)  # [1,1,-2]/√6
    # Columns are the file-frame basis vectors written in cubic coords
    U = np.column_stack([ex, ey, ez])  # maps components in 111-frame -> cubic
    return U


def A_file_to_cubic(A_file):
    U = U_111_to_cubic()
    return U @ A_file @ U.T


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
