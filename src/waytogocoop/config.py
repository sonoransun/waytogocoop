"""Physical constants and default parameters for moire CPDM calculations."""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
HBAR_EV_S: float = 6.582119569e-16          # hbar in eV*s
KB_EV_K: float = 8.617333262e-5             # Boltzmann constant in eV/K
ELECTRON_MASS_KG: float = 9.1093837015e-31  # free electron mass

# ---------------------------------------------------------------------------
# Superconducting gap parameters (meV)
# ---------------------------------------------------------------------------
DELTA_1: float = 2.58   # smaller gap observed in topological surface states
DELTA_2: float = 3.60   # larger gap / bulk-like gap
DELTA_AVG: float = (DELTA_1 + DELTA_2) / 2.0
DELTA_AMPLITUDE: float = (DELTA_2 - DELTA_1) / 2.0

# ---------------------------------------------------------------------------
# Default computation grid
# ---------------------------------------------------------------------------
DEFAULT_GRID_SIZE: int = 200         # number of grid points per axis
DEFAULT_PHYSICAL_EXTENT: float = 100.0  # Angstrom — half-width of the real-space window
DEFAULT_TWIST_ANGLE: float = 0.0    # degrees

# ---------------------------------------------------------------------------
# Coherence length (Angstrom) — BCS estimate for FeTe-based systems
# ---------------------------------------------------------------------------
DEFAULT_COHERENCE_LENGTH: float = 20.0

# ---------------------------------------------------------------------------
# FFT peak detection
# ---------------------------------------------------------------------------
DEFAULT_FFT_THRESHOLD_FRACTION: float = 0.3

# ---------------------------------------------------------------------------
# Isotope-effect parameters (speculative)
# ---------------------------------------------------------------------------
AMU_TO_KG: float = 1.66053906660e-27       # atomic mass unit -> kg
HBAR_J_S: float = 1.054571817e-34          # hbar in J*s
DEFAULT_ISOTOPE_EXPONENT: float = 0.25     # BCS isotope exponent for FeTe (anomalous)

# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------
ANGSTROM_TO_NM: float = 0.1
NM_TO_ANGSTROM: float = 10.0
MEV_TO_EV: float = 1.0e-3
