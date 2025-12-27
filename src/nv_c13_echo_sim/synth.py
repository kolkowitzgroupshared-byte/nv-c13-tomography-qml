# src/utils/synth.py
from __future__ import annotations

import inspect
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, Optional, Tuple

from nv_c13_echo_sim import simulate_random_spin_echo, plot_echo_with_sites
from nv_c13_echo_sim.priors import nv_prior_draw

def _call_sim(
    *,
    hyperfine_path: str,
    tau_range_us: Tuple[float, float],
    num_spins: Optional[int],
    distance_cutoff: float,
    f_range_kHz: Tuple[float, float],
    selection_mode: str,
    abundance_fraction: float,
    reuse_present_mask: bool,
    fine_params: Dict[str, Any],
    nv_orientation: Any,
    catalog_json_path: Optional[str],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Call simulate_random_spin_echo() but adapt to whichever keyword names exist
    in your installed nv_c13_tomography version.
    """
    Ak_min_kHz, Ak_max_kHz = f_range_kHz

    sig = inspect.signature(simulate_random_spin_echo)
    P = sig.parameters

    kw: Dict[str, Any] = {}

    # required-ish common args
    kw["hyperfine_path"] = hyperfine_path
    kw["tau_range_us"] = tau_range_us
    kw["num_spins"] = num_spins
    kw["selection_mode"] = selection_mode
    kw["abundance_fraction"] = abundance_fraction
    kw["reuse_present_mask"] = reuse_present_mask
    kw["fine_params"] = fine_params
    kw["nv_orientation"] = nv_orientation
    kw["rng_seed"] = 4242

    # optional catalog (if supported)
    kw["catalog_json_path"] = catalog_json_path

    # num realizations (name differs sometimes)
    if "num_realizations" in P:
        kw["num_realizations"] = 1
    elif "num_realization" in P:
        kw["num_realization"] = 1

    # distance cutoff keyword differs
    if "distance_cutoff" in P:
        kw["distance_cutoff"] = distance_cutoff
    elif "distance_cutoff_A" in P:
        kw["distance_cutoff_A"] = distance_cutoff

    # frequency band keyword differs
    if "f_band_kHz" in P:
        kw["f_band_kHz"] = (Ak_min_kHz, Ak_max_kHz)
    elif "f_range_kHz" in P:
        kw["f_range_kHz"] = (Ak_min_kHz, Ak_max_kHz)
    elif ("Ak_min_kHz" in P) and ("Ak_max_kHz" in P):
        kw["Ak_min_kHz"] = Ak_min_kHz
        kw["Ak_max_kHz"] = Ak_max_kHz

    # finally: drop anything the function doesn't accept
    kw = {k: v for k, v in kw.items() if (k in P and v is not None)}

    return simulate_random_spin_echo(**kw)


def synth_per_nv(
    *,
    nv_idx: int,
    labels: list,
    nv_orientations: list,
    P: np.ndarray,
    K: dict,
    cohort_med: np.ndarray,
    cohort_mad: np.ndarray,
    R: int = 1,
    tau_range_us: Tuple[float, float] = (0, 100),
    f_range_kHz: Tuple[float, float] = (10, 1500),
    hyperfine_path: str = "analysis/nv_hyperfine_coupling/nv-2.txt",
    abundance_fraction: float = 0.011,
    distance_cutoff: float = 22.0,
    num_spins: Optional[int] = 1,
    selection_mode: str = "top_kappa",
    reuse_present_mask: bool = True,
    catalog_json_path: Optional[str] = None,
    rng_seed: int = 20251102,
    mix_global: float = 0.3,
    show_each_plot: bool = True,
) -> Dict[str, Any]:
    lbl = int(labels[nv_idx])
    nv_ori = nv_orientations[nv_idx]

    rng = np.random.default_rng(int(rng_seed))

    traces = []
    fine_list = []
    taus_ref = None

    aux_first = None
    fine_first = None
    echo_first = None

    for r in range(int(R)):
        fp = nv_prior_draw(
            nv_idx=nv_idx,
            P=P,
            K=K,
            cohort_med=cohort_med,
            cohort_mad=cohort_mad,
            rng=rng,
            mix_global=mix_global,
        )

        fine_params = dict(
            baseline=fp["baseline"],
            comb_contrast=fp["comb_contrast"],
            revival_time_us=fp["revival_time_us"],
            width0_us=fp["width0_us"],
            T2_ms=fp["T2_ms"],
            T2_exp=fp["T2_exp"],
            amp_taper_alpha=fp["amp_taper_alpha"],
            width_slope=fp["width_slope"],
            revival_chirp=fp["revival_chirp"],
            osc_amp=fp["osc_amp"],
            osc_f0=fp["osc_f0"],
            osc_f1=fp["osc_f1"],
            osc_phi0=fp["osc_phi0"],
            osc_phi1=fp["osc_phi1"],
        )

        taus, echo, aux = _call_sim(
            hyperfine_path=hyperfine_path,
            tau_range_us=tau_range_us,
            num_spins=num_spins,
            distance_cutoff=distance_cutoff,
            f_range_kHz=f_range_kHz,
            selection_mode=selection_mode,
            abundance_fraction=abundance_fraction,
            reuse_present_mask=reuse_present_mask,
            fine_params=fine_params,
            nv_orientation=nv_ori,
            catalog_json_path=catalog_json_path,
        )

        plot_echo_with_sites(
            taus,
            echo,
            aux,
            fine_params=fine_params,
            nv_label=lbl,
            nv_orientation=nv_ori,
            sim_info=dict(
                selection_mode=selection_mode,
                distance_cutoff=distance_cutoff,
                Ak_min_kHz=f_range_kHz[0],
                Ak_max_kHz=f_range_kHz[1],
                reuse_present_mask=reuse_present_mask,
                hyperfine_path=hyperfine_path,
                abundance_fraction=abundance_fraction,
                num_spins=("all" if num_spins is None else int(num_spins)),
            ),
            show_env=False,
            show_env_times_comb=False,
        )

        if show_each_plot:
            plt.show()

        if taus_ref is None:
            taus_ref = np.asarray(taus, float)

        traces.append(np.asarray(echo, float))
        fine_list.append(fine_params)

        if r == 0:
            aux_first = aux
            fine_first = fine_params
            echo_first = np.asarray(echo, float)

    traces = np.asarray(traces, float)
    mean = np.nanmean(traces, axis=0)
    p16 = np.nanpercentile(traces, 16, axis=0)
    p84 = np.nanpercentile(traces, 84, axis=0)

    return dict(
        label=lbl,
        nv_orientation=nv_ori,
        taus_us=taus_ref,
        mean=mean,
        p16=p16,
        p84=p84,
        fine_draws=fine_list,
        first_echo=echo_first,
        aux_first=aux_first,
        fine_params_first=fine_first,
    )
