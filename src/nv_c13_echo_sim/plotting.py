from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict

# =============================================================================
# Plotting
# =============================================================================
def set_axes_equal_3d(ax):
    """Make 3D axes have equal scale (so spheres look like spheres)."""
    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()
    xmid = 0.5 * (xlim[0] + xlim[1])
    ymid = 0.5 * (ylim[0] + ylim[1])
    zmid = 0.5 * (zlim[0] + zlim[1])
    max_range = 0.5 * max(xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0])
    ax.set_xlim3d(xmid - max_range, xmid + max_range)
    ax.set_ylim3d(ymid - max_range, ymid + max_range)
    ax.set_zlim3d(zmid - max_range, zmid + max_range)

def _echo_summary_lines(taus_us, echo):
    if len(echo) == 0:
        return []
    arr = np.asarray(echo, float)
    n = max(3, len(arr) // 3)
    early = float(np.nanmean(arr[:n]))
    late = float(np.nanmean(arr[-n:]))
    return [
        f"Echo range: {arr.min():.3f} … {arr.max():.3f}",
        f"⟨early⟩→⟨late⟩: {early:.3f} → {late:.3f}",
    ]

def _fine_param_lines(fine_params):
    if not fine_params:
        return []
    pretty = {
        "revival_time": "T_rev (μs)",
        "width0_us": "width₀ (μs)",
        "T2_ms": "T₂ (ms)",
        "T2_exp": "stretch n",
        "amp_taper_alpha": "amp taper α",
        "width_slope": "width slope",
        "revival_chirp": "rev chirp",
    }
    keys = [
        "revival_time",
        "width0_us",
        "T2_ms",
        "T2_exp",
        "amp_taper_alpha",
        "width_slope",
        "revival_chirp",
    ]
    out = []
    for k in keys:
        if k in fine_params:
            v = fine_params[k]
            sval = f"{v:.3g}" if isinstance(v, (int, float)) else f"{v}"
            out.append(f"{pretty[k]}: {sval}")
    return out

#
def _site_table_lines(site_info, max_rows=8):
    """
    Build a small text table for the annotation box on the 3D panel.

    For catalog mode we show:
      site_id | r (Å) | f_- (kHz) | f_+ (kHz)
    """
    if not site_info:
        return ["No sites selected"]

    lines = []
    lines.append("id   r(Å)   f_-(kHz)   f_+(kHz)")

    n = min(len(site_info), max_rows)
    for meta in site_info[:n]:
        sid = meta.get("site_id", "?")
        r = float(meta.get("r", np.nan))
        f_m = float(meta.get("f_minus_kHz", np.nan))
        f_p = float(meta.get("f_plus_kHz", np.nan))

        lines.append(f"{sid:3d}  {r:5.2f}   {f_m:8.1f}   {f_p:8.1f}")

    if len(site_info) > max_rows:
        lines.append(f"... (+{len(site_info) - max_rows} more)")

    return lines


def _env_only_curve(taus_us, fine_params):
    """baseline - envelope(τ); ignores COMB/MOD so you see pure T2 envelope."""
    if not fine_params:
        return None
    baseline = float(fine_params.get("baseline", 1.0))
    T2_ms = float(fine_params.get("T2_ms", 1.0))
    T2_exp = float(fine_params.get("T2_exp", 1.0))
    # envelope(τ) = exp[-(τ/(1000*T2_ms))^T2_exp]
    env = np.exp(-((np.asarray(taus_us, float) / (1000.0 * T2_ms)) ** T2_exp))
    # multiply by comb_contrast if you want to visualize the amplitude scale
    contrast = float(fine_params.get("comb_contrast", 1.0))
    return baseline - contrast * env


def _comb_only_curve(taus_us, fine_params):
    """
    Very light-weight comb sketch (Gaussian revivals); ignores oscillations and width slope.
    Useful if you want to also show envelope×comb (set show_env_times_comb=True).
    """
    if not fine_params:
        return None
    T_rev = float(
        fine_params.get("revival_time", fine_params.get("revival_time_us", 0.0))
    )
    width0 = float(fine_params.get("width0_us", 0.0))
    alpha = float(fine_params.get("amp_taper_alpha", 0.0))
    if T_rev <= 0 or width0 <= 0:
        return np.ones_like(taus_us, dtype=float)

    τ = np.asarray(taus_us, float)
    mmax = int(max(1, np.ceil(τ.max() / T_rev) + 2))
    comb = np.zeros_like(τ, float)
    # sum of Gaussians centered at m*T_rev with amplitude taper ~ exp(-alpha*m)
    for m in range(mmax + 1):
        amp = np.exp(-alpha * m) if alpha > 0 else 1.0
        comb += amp * np.exp(-0.5 * ((τ - m * T_rev) / width0) ** 2)
    # normalize to [0,1] peak
    mx = comb.max()
    if mx > 0:
        comb = comb / mx
    return comb


def plot_echo_with_sites(
    taus_us,
    echo,
    aux,
    title="Spin Echo (single NV)",
    rmax=None,
    fine_params=None,
    units_label="(arb units)",
    nv_label=None,  # <-- NEW: show NV id
    nv_orientation=None,
    sim_info=None,  # <-- NEW: dict with sim settings to display
    show_env=True,  # <-- NEW: overlay envelope-only
    show_env_times_comb=False,  # <-- NEW: optionally overlay envelope×comb
):
    fig = plt.figure(figsize=(12, 5))

    # ---------------- Echo panel ----------------
    ax0 = fig.add_subplot(1, 2, 1)
    ax0.plot(taus_us, echo, lw=1.0, label="echo")
    ax0.set_xlabel("τ (μs)")
    ax0.set_ylabel(f"Coherence {units_label}")

    # Title: include NV label if provided
    if nv_label and nv_orientation is not None:
        ax0.set_title(f"{title} — NV {nv_label} [{nv_orientation}]")
    elif nv_label is not None and nv_orientation is None:
        ax0.set_title(f"{title} — NV {nv_label}")
    else:
        ax0.set_title(title)

    ax0.grid(True, alpha=0.3)

    # Vertical revival guide lines (if provided)
    revs = aux.get("revivals_us", None)
    if revs is not None:
        for t in np.atleast_1d(revs):
            ax0.axvline(t, ls="--", lw=0.7, alpha=0.35)

    # --- NEW: overlay envelope(s) ---
    env_line = None
    if show_env and fine_params:
        y_env = _env_only_curve(taus_us, fine_params)
        if y_env is not None:
            (env_line,) = ax0.plot(
                taus_us, y_env, ls="--", lw=1.2, label="envelope (T₂)", alpha=0.9
            )

    if show_env_times_comb and fine_params:
        comb = _comb_only_curve(taus_us, fine_params)
        if comb is not None:
            baseline = float(fine_params.get("baseline", 1.0))
            contrast = float(fine_params.get("comb_contrast", 1.0))
            T2_ms = float(fine_params.get("T2_ms", 1.0))
            T2_exp = float(fine_params.get("T2_exp", 1.0))
            env = np.exp(-((np.asarray(taus_us, float) / (1000.0 * T2_ms)) ** T2_exp))
            y_env_comb = baseline - contrast * env * comb
            ax0.plot(
                taus_us,
                y_env_comb,
                ls=":",
                lw=1.2,
                label="envelope×comb (no osc)",
                alpha=0.9,
            )

    # Existing stats box
    stats = aux.get("stats", {}) or {}
    # lines_stats = []
    # if "N_candidates" in stats:
    #     lines_stats.append(f"Candidates: {stats['N_candidates']}")
    # if "abundance_fraction" in stats:
    #     lines_stats.append(f"Abundance p: {100*stats['abundance_fraction']:.2f}%")
    # if "chosen_counts" in stats and stats["chosen_counts"]:
    #     cc = np.asarray(stats["chosen_counts"], int)
    #     lines_stats.append(f"Chosen/site per realization: {int(np.median(cc))} (med)")
    # if lines_stats:
    #     ax0.text(
    #         0.61,
    #         0.02,
    #         "\n".join(lines_stats),
    #         transform=ax0.transAxes,
    #         fontsize=9,
    #         va="bottom",
    #         ha="left",
    #         bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.6, lw=0.6),
    #     )

    # Fine-parameter box (existing)
    # ---- Combined NV/sim + fine-params box (single box, right-top) ----
    combined_lines = []

    # Header & flags
    if nv_label is not None:
        flag_bits = []
        if show_env:
            flag_bits.append("Env")
        if show_env_times_comb:
            flag_bits.append("Comb")
        hdr = f"NV: {nv_label}"
        if flag_bits:
            hdr += f"  [{'+'.join(flag_bits)} shown]"
        combined_lines.append(hdr)

    # Build a meta dict from sim_info with fallbacks to aux
    meta = {} if sim_info is None else dict(sim_info)
    # meta.setdefault("selection_mode", aux.get("selection_mode"))
    meta.setdefault("distance_cutoff", aux.get("distance_cutoff"))
    meta.setdefault("Ak_min_kHz", aux.get("Ak_min_kHz"))
    meta.setdefault("Ak_max_kHz", aux.get("Ak_max_kHz"))
    # meta.setdefault("Ak_abs", aux.get("Ak_abs"))
    # meta.setdefault("reuse_present_mask", aux.get("reuse_present_mask"))
    # meta.setdefault("hyperfine_path", aux.get("hyperfine_path"))
    meta.setdefault("T2_fit_us", None)  # you can set this upstream if desired

    # Pretty labels
    pretty_sim = {
        # "selection_mode": "select",
        "distance_cutoff": "d_cut (Å)",
        "Ak_min_kHz": "Ak_min (kHz)",
        "Ak_max_kHz": "Ak_max (kHz)",
        # "Ak_abs": "Ak|·|?",
        # "reuse_present_mask": "reuse mask?",
        # "hyperfine_path": "HF",
        "T2_fit_us": "T2_fit (μs)",
    }

    def _fmt_meta(k, v):
        if v is None:
            return None
        lab = pretty_sim.get(k, k)
        if k == "hyperfine_path":
            from pathlib import Path

            v = Path(str(v)).stem
        if isinstance(v, float):
            # compact floats
            v = f"{v:.3g}"
        return f"{lab}: {v}"

    # Collect sim/meta lines (only those that exist)
    sim_lines = []
    for k in [
        # "selection_mode",
        "distance_cutoff",
        "Ak_min_kHz",
        "Ak_max_kHz",
        # "Ak_abs",
        # "reuse_present_mask",
        # "hyperfine_path",
        "T2_fit_us",
    ]:
        line = _fmt_meta(k, meta.get(k))
        if line:
            sim_lines.append(line)

    # Fine-parameter lines
    fp_lines = _fine_param_lines(fine_params) if fine_params else []
    if fp_lines and show_env:
        fp_lines = ["Exp Params."] + fp_lines

    # Merge sections with a thin separator if both present
    if sim_lines and fp_lines:
        combined_lines.extend(sim_lines + ["—"] + fp_lines)
    elif sim_lines:
        combined_lines.extend(sim_lines)
    elif fp_lines:
        combined_lines.extend(fp_lines)

    # Render the single box (right-top)
    # if combined_lines:
    #     ax0.text(
    #         0.99,
    #         0.02,
    #         "\n".join(combined_lines),
    #         transform=ax0.transAxes,
    #         fontsize=9,
    #         va="bottom",
    #         ha="right",
    #         bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.5, lw=0.6),
    #     )

    # Legend if we drew extra curves
    if (show_env and fine_params) or show_env_times_comb:
        ax0.legend(loc="best", fontsize=9, framealpha=0.8)

    # ---------------- 3D positions panel ----------------
    ax1 = fig.add_subplot(1, 2, 2, projection="3d")
    bg = aux.get("all_candidate_positions", None)
    if bg is not None and len(bg) > 0:
        ax1.scatter(bg[:, 0], bg[:, 1], bg[:, 2], s=8, alpha=0.15)

    pos = aux.get("positions", None)
    info = aux.get("site_info", [])
    if pos is not None and len(pos) > 0:
        ax1.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=20, depthshade=True)
        # for pnt, meta in zip(pos, info):
        #     # sid = meta.get("site_id", "?")
        #     # apar = meta.get("Apar_kHz", 0.0)
        #     rmag = meta.get("r", np.nan)
        #     # label = f"{sid}\n|A∥|={apar:.0f} kHz\nr={rmag:.2f}"
        #     label = f"r={rmag:.2f}"
        #     # label = f"{sid}"
        #     ax1.text(pnt[0], pnt[1], pnt[2], label, fontsize=8, ha="left", va="bottom")
        for pnt, meta in zip(pos, info):
            rmag = meta.get("r", np.nan)
            f_m = meta.get("f_minus_kHz", np.nan)
            label = f"r={rmag:.2f}\nf_-={f_m:.0f} kHz"
            ax1.text(pnt[0], pnt[1], pnt[2], label, fontsize=7, ha="left", va="bottom")

    ax1.scatter([0], [0], [0], s=70, marker="*", zorder=5)
    ax1.text(0, 0, 0, "NV", fontsize=9, ha="right", va="top")
    ax1.set_title("¹³C positions (NV frame)")
    ax1.set_xlabel("x (Å)")
    ax1.set_ylabel("y (Å)")
    ax1.set_zlabel("z (Å)")

    if rmax is None:
        if bg is not None and len(bg) > 0:
            rmax = float(np.max(np.linalg.norm(bg, axis=1)))
        elif pos is not None and len(pos) > 0:
            rmax = float(np.max(np.linalg.norm(pos, axis=1)))
        else:
            rmax = 1.0
    rpad = 0.05 * rmax
    ax1.set_xlim(-rmax - rpad, rmax + rpad)
    ax1.set_ylim(-rmax - rpad, rmax + rpad)
    ax1.set_zlim(-rmax - rpad, rmax + rpad)
    set_axes_equal_3d(ax1)

    picked_all = aux.get("picked_ids_per_realization", [])
    n_real = len(picked_all) if picked_all is not None else 0
    n_chosen = len(info) if info is not None else 0
    left_box = [f"Chosen Sites: {n_chosen}", f"Realizations: {n_real}"]
    if stats.get("N_candidates") is not None:
        left_box.append(f"Candidates: {stats['N_candidates']}")
    if "abundance_fraction" in stats:
        left_box.append(f"Abundance p: {100*stats['abundance_fraction']:.2f}%")
    ax1.text2D(
        0.01,
        0.02,
        "\n".join(left_box),
        transform=ax1.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.8, lw=0.6),
    )

    table_lines = _site_table_lines(info, max_rows=8)
    ax1.text2D(
        0.99,
        0.02,
        "\n".join(table_lines),
        transform=ax1.transAxes,
        fontsize=9,
        # family="monospace",
        va="bottom",
        ha="right",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.8, lw=0.6),
    )

    # kpl.show()
    return fig
