from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List
import json
from pathlib import Path


@dataclass(frozen=True)
class CatalogSite:
    site_id: int
    orientation: tuple[int, int, int] | None
    distance_A: float | None

    kappa: float
    f0_Hz: float        # fI_Hz
    f_m1_Hz: float      # omega_ms_Hz

    f_minus_Hz: float
    f_plus_Hz: float

    line_w_minus: float = 0.0
    line_w_plus: float = 0.0

    x_A: float | None = None
    y_A: float | None = None
    z_A: float | None = None


def load_catalog(path: str | Path) -> List[CatalogSite]:
    path = Path(path)
    data: List[Dict[str, Any]] = json.loads(path.read_text())
    sites: List[CatalogSite] = []
    for rec in data:
        ori = rec.get("orientation", None)
        sites.append(
            CatalogSite(
                site_id=int(rec["site_index"]),
                orientation=tuple(int(x) for x in ori) if ori is not None else None,
                distance_A=float(rec["distance_A"]) if "distance_A" in rec else None,
                kappa=float(rec["kappa"]),
                f0_Hz=float(rec["fI_Hz"]),
                f_m1_Hz=float(rec["omega_ms_Hz"]),
                f_minus_Hz=float(rec["f_minus_Hz"]),
                f_plus_Hz=float(rec["f_plus_Hz"]),
                line_w_minus=float(rec.get("line_w_minus", 0.0)),
                line_w_plus=float(rec.get("line_w_plus", 0.0)),
                x_A=(float(rec["x_A"]) if "x_A" in rec else None),
                y_A=(float(rec["y_A"]) if "y_A" in rec else None),
                z_A=(float(rec["z_A"]) if "z_A" in rec else None),
            )
        )
    return sites
