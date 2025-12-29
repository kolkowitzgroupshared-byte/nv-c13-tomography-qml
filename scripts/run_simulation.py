from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml
import matplotlib.pyplot as plt

from nv_c13_echo_sim import build_param_matrix_from_fit, cohort_median_mad
from nv_c13_echo_sim import select_nvs
from nv_c13_echo_sim import synth_per_nv


def _read_text_strip_bom(p: Path) -> str:
    return p.read_text(encoding="utf-8-sig")


def _load_yaml(p: Path) -> Dict[str, Any]:
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p.resolve()}")
    return yaml.safe_load(_read_text_strip_bom(p))


def _resolve_path(maybe_path: Any, base_dir: Path) -> Optional[Path]:
    if maybe_path is None:
        return None
    s = str(maybe_path).strip()
    if not s:
        return None
    p = Path(s)
    return p if p.is_absolute() else (base_dir / p)


def _require_exists(label: str, maybe_path: Any, base_dir: Path) -> Optional[str]:
    p = _resolve_path(maybe_path, base_dir)
    if p is None:
        return None
    if not p.exists():
        raise FileNotFoundError(f"{label} not found: {p.resolve()}")
    return str(p)


def _maybe_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def _load_fit_from_file(fits_path: str) -> Dict[str, Any]:
    p = Path(fits_path)
    try:
        return json.loads(p.read_text())
    except Exception:
        return json.loads(_read_text_strip_bom(p))


def main() -> List[Dict[str, Any]]:
    repo_root = Path(__file__).resolve().parents[1]  # repo/
    cfg_path = repo_root / "configs" / "batch.yaml"  # <- default, no CLI
    cfg = _load_yaml(cfg_path)

    paths_cfg = cfg.get("paths", {}) or {}
    batch_cfg = cfg.get("batch", {}) or {}
    sim_cfg = cfg.get("sim", {}) or {}
    plot_cfg = cfg.get("plot", {}) or {}

    # inputs
    hyperfine_path = _require_exists("hyperfine_path", paths_cfg.get("hyperfine_path"), repo_root)
    catalog_path = _require_exists("catalog_path", paths_cfg.get("catalog_path"), repo_root)
    fits_path = _require_exists("fits_path", paths_cfg.get("fits_path"), repo_root)

    # outputs
    out_dir = _resolve_path(paths_cfg.get("out_dir", "outputs/sim_batch"), repo_root)
    assert out_dir is not None
    out_prefix = str(paths_cfg.get("out_prefix", "sim"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # load fits
    if fits_path is None:
        raise ValueError("paths.fits_path is required.")
    fit = _load_fit_from_file(fits_path)

    pack = build_param_matrix_from_fit(fit)
    cohort_med, cohort_mad = cohort_median_mad(pack["P"])

    # select NVs
    keep = select_nvs(
        P=pack["P"],
        K=pack["K"],
        popts=pack["popts"],
        chis=pack["chis"],
        max_keep=int(batch_cfg.get("num_spectra", 15)),
        contrast_min=float(batch_cfg.get("contrast_min", 0.10)),
        contrast_max=float(batch_cfg.get("contrast_max", 2.0)),
        t2_threshold_us=float(batch_cfg.get("t2_threshold_us", 2000.0)),
    )
    print(f"Selected {len(keep)} NVs (spectra).")

    # sim params (only those synth_per_nv supports)
    tau_range_us = tuple(sim_cfg.get("tau_range_us", [0.0, 100.0]))
    f_range_kHz = tuple(sim_cfg.get("f_band_kHz", [10.0, 1500.0]))
    distance_cutoff = float(sim_cfg.get("distance_cutoff_A", 22.0))  # synth expects distance_cutoff
    abundance_fraction = float(sim_cfg.get("abundance_fraction", 0.011))
    num_spins = sim_cfg.get("num_spins", 1)
    num_spins = None if num_spins is None else int(num_spins)
    selection_mode = str(sim_cfg.get("selection_mode", "uniform"))
    reuse_present_mask = bool(sim_cfg.get("reuse_present_mask", True))
    R = int(batch_cfg.get("R", 1))
    rng_seed = int(batch_cfg.get("rng_seed", 424242))

    save_per_nv = bool(plot_cfg.get("save_per_nv", False))
    show_each_plot = bool(plot_cfg.get("show_each", True))

    results: List[Dict[str, Any]] = []
    for idx in keep:
        res = synth_per_nv(
            nv_idx=int(idx),
            labels=pack["labels"],
            nv_orientations=pack["nv_orientations"],
            P=pack["P"],
            K=pack["K"],
            cohort_med=cohort_med,
            cohort_mad=cohort_mad,
            R=R,
            tau_range_us=tau_range_us,
            f_range_kHz=f_range_kHz,
            hyperfine_path=hyperfine_path,
            catalog_json_path=catalog_path,
            abundance_fraction=abundance_fraction,
            distance_cutoff=distance_cutoff,
            num_spins=num_spins,
            selection_mode=selection_mode,
            reuse_present_mask=reuse_present_mask,
            rng_seed=rng_seed,
            show_each_plot=show_each_plot,
        )
        results.append(res)

        if save_per_nv:
            nv_label = int(res["label"])
            out_json = out_dir / f"{out_prefix}_NV{nv_label:03d}.json"
            _maybe_write_json(
                out_json,
                dict(
                    label=nv_label,
                    nv_orientation=list(res["nv_orientation"]),
                    taus_us=np.asarray(res["taus_us"]).tolist(),
                    mean=np.asarray(res["mean"]).tolist(),
                    p16=np.asarray(res["p16"]).tolist() if res.get("p16") is not None else None,
                    p84=np.asarray(res["p84"]).tolist() if res.get("p84") is not None else None,
                ),
            )

    # overview plot
    plt.figure(figsize=(8, 5))
    for r in results:
        plt.plot(r["taus_us"], r["mean"], lw=1.0, alpha=0.9)
    plt.xlabel("τ (μs)")
    plt.ylabel("echo (mean)")
    plt.title("All simulated NV spectra (mean)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # out_png = out_dir / f"{out_prefix}_overview.png"
    # plt.savefig(out_png, dpi=200, bbox_inches="tight")
    # print(f"Saved overview plot: {out_png.resolve()}")
    plt.show()

    return results


if __name__ == "__main__":
    main()
