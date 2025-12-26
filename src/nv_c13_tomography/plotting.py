from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict


def plot_echo(taus_us: np.ndarray, echo: np.ndarray, title: str = "NV 13C echo", aux: Optional[Dict] = None):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(taus_us, echo, lw=1.2)
    ax.set_xlabel("τ (μs)")
    ax.set_ylabel("Coherence (arb.)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if aux:
        text = "\n".join([f"{k}: {v}" for k, v in aux.items() if k in ("N_candidates","N_present","N_chosen")])
        ax.text(
            0.98, 0.02, text,
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )
    return fig
