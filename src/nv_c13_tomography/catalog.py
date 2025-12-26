from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict, Any
import json
from pathlib import Path

@dataclass(frozen=True)
class CatalogSite:
    site_id: int
    orientation: Tuple[int, int, int]
    distance_A: Optional[float]
    kappa: float
    f0_Hz: float
    f_m1_Hz: float
    f_minus_Hz: float
    f_plus_Hz: float
    x_A: Optional[float] = None
    y_A: Optional[float] = None
    z_A: Optional[float] = None
    line_w_minus: float = 0.0
    line_w_plus: float = 0.0

def load_catalog(path_json: str | Path) -> List[Dict[str, Any]]:
    return json.loads(Path(path_json).read_text())
