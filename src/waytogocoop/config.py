"""Physical constants and default parameters for moire CPDM calculations."""

from __future__ import annotations

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
DEFAULT_ISOTOPE_EXPONENT: float = 0.4      # BCS isotope exponent — literature consensus
# Literature range: α = 0.35–0.8 for iron chalcogenides (Khasanov 2010, Liu 2009);
# α = −0.18 inverse effect in (Ba,K)Fe₂As₂ (Shirage 2009).
# 0.4 is the corrected consensus value (PRB 82, 212505).

# ---------------------------------------------------------------------------
# Magnetic / topological constants
# ---------------------------------------------------------------------------
PHI_0: float = 2.0678e-15              # Magnetic flux quantum h/(2e) in Wb
MU_B_EV_T: float = 5.788e-5            # Bohr magneton in eV/T
MU_B_MEV_T: float = 5.788e-2           # Bohr magneton in meV/T
V_F_TI: float = 5.0e5                  # TI surface Dirac fermion velocity (m/s)
G_FACTOR_TSS: float = 30.0             # Effective g-factor for topological surface states
                                        # Literature range: 20–50 (Fu & Kane, PRL 2008)
LAMBDA_L_FETE: float = 5000.0          # London penetration depth for FeTe (Angstrom)
                                        # ~500 nm typical for iron chalcogenides
XI_PROXIMITY_DEFAULT: float = 100.0    # Proximity coherence length into TI (Angstrom)
                                        # Literature: 5–20 nm → 50–200 Angstrom
XI_MAJORANA_DEFAULT: float = 50.0      # Majorana localization length (Angstrom), speculative
K_F_TSS: float = 0.1                   # TI surface Fermi wavevector (1/Angstrom)
                                        # Approximate; depends on Fermi level tuning
BC2_FETE: float = 47.0                 # Upper critical field Hc2 for FeTe (Tesla)
                                        # Literature: ~47 T at low temperature
ANGSTROM_TO_M: float = 1.0e-10         # Angstrom → metre conversion

# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------
ANGSTROM_TO_NM: float = 0.1

# ---------------------------------------------------------------------------
# Numerical thresholds
# ---------------------------------------------------------------------------
ZERO_THRESHOLD: float = 1e-12          # Float-equality-to-zero checks
SMALL_ANGLE_THRESHOLD: float = 1e-6   # Degrees; below this, twist is treated as zero
NORM_FLOOR: float = 1e-15             # Minimum denominator for normalization
EXPONENT_CLAMP: float = 100.0         # Max |exponent| for exp() overflow protection
PEAK_POWER_FLOOR: float = 1e-30       # Below this, power spectrum is treated as zero
