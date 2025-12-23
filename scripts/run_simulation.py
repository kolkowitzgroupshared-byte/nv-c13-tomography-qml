from __future__ import annotations
import argparse
from pathlib import Path
import yaml

from nv_c13_tomography.catalog import load_catalog
from nv_c13_tomography.simulator import simulate_catalog_mode
from nv_c13_tomography.plotting import plot_echo
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    catalog = load_catalog(cfg["catalog_path"])

    res = simulate_catalog_mode(
        catalog=catalog,
        tau_range_us=tuple(cfg["tau_range_us"]),
        n_tau=int(cfg.get("n_tau", 600)),
        B_vec_G=tuple(cfg["B_vec_G"]),
        abundance_fraction=float(cfg.get("abundance_fraction", 0.011)),
        num_spins=cfg.get("num_spins", None),
        selection_mode=str(cfg.get("selection_mode", "top_kappa")),
        rng_seed=int(cfg.get("rng_seed", 1234)),
        fine_params=cfg.get("fine_params", None),
        nv_orientation=tuple(cfg["nv_orientation"]) if cfg.get("nv_orientation") else None,
        distance_cutoff_A=cfg.get("distance_cutoff_A", None),
        f_band_kHz=tuple(cfg["f_band_kHz"]) if cfg.get("f_band_kHz") else None,
        reuse_present_mask=bool(cfg.get("reuse_present_mask", True)),
    )

    fig = plot_echo(res.taus_us, res.echo, title=cfg.get("title", "NV 13C Tomography"), aux=res.aux)
    out = Path(cfg.get("output_png", "outputs/echo.png"))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
