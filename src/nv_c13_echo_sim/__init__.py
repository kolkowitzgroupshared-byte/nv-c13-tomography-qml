# __init__.py
from .simulator import simulate_random_spin_echo
from .plotting import plot_echo_with_sites
from .noise import NoiseSpec, apply_noise_pipeline
from .catalog import load_catalog
from .synth import synth_per_nv
from .rng import spawn_streams, _choose_sites
from .priors import nv_prior_draw
from .selection import select_nvs
from .fit_loader import build_param_matrix_from_fit, cohort_median_mad



