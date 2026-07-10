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

# HIGHLY SPECULATIVE exploration ranges (amu) for the exotic-isotope mode:
# masses span far beyond the known isotopes, out towards the driplines.
# Values between/beyond known isotopes are purely hypothetical what-ifs.
EXOTIC_MASS_RANGES: dict[str, tuple[float, float]] = {
    "Fe": (45.0, 75.0),
    "Te": (105.0, 145.0),
    "Sb": (103.0, 140.0),
    "C": (8.0, 22.0),
}

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

# ---------------------------------------------------------------------------
# Graphene / twisted-graphene parameters
# ---------------------------------------------------------------------------
GRAPHENE_A: float = 2.46          # Angstrom — graphene in-plane lattice constant
GRAPHENE_A_CC: float = 1.42       # Angstrom — C-C bond length, a/sqrt(3)
HBAR_VF_GRAPHENE: float = 5.96    # eV*Angstrom — hbar*v_F = (sqrt(3)/2)*t*a with t = 2.8 eV
                                  # NN hopping (Castro Neto et al., RMP 81, 109 (2009));
                                  # v_F ~ 0.91e6 m/s.  NOT the common 6.58 value — with 6.58
                                  # the first magic angle lands at 0.974 deg, outside the
                                  # observed 1.0-1.2 deg window.
W_INTERLAYER_TBG: float = 0.110   # eV — AB interlayer tunneling (Bistritzer & MacDonald,
                                  # PNAS 108, 12233 (2011)); with HBAR_VF_GRAPHENE gives a
                                  # first magic angle of 1.076 deg, matching the repo's
                                  # 1.08-deg TBG tests
BCS_GAP_RATIO: float = 1.764      # Delta(0)/(kB*Tc), weak-coupling BCS
DELTA_TBG_MAX: float = 0.30       # meV — ~1.764*kB*Tc for Tc~2 K (Cao et al., Nature 556,
                                  # 43 (2018)); speculative model input
DELTA_TTG_MAX: float = 0.44       # meV — Tc~2.9 K alternating-twist trilayer (Park et al.,
                                  # Nature 590, 249 (2021)); speculative model input
THETA_SC_WIDTH_DEG: float = 0.1   # deg — Lorentzian HWHM of gap vs twist; SPECULATIVE —
                                  # SC observed roughly 0.9-1.2 deg
NU_OPTIMAL_FILLING: float = 2.4   # electrons per moire cell (Cao 2018)
NU_DOME_WIDTH: float = 0.8        # SPECULATIVE dome half-width in filling
XI_TBG: float = 500.0             # Angstrom — GL coherence length ~52 nm (Cao 2018)
GRAPHENE_EXTENT_DEFAULT: float = 200.0  # Angstrom half-width; ~3 moire periods at magic angle

# ---------------------------------------------------------------------------
# Curved-sheet pseudo-magnetic-field parameters (speculative)
# ---------------------------------------------------------------------------
ELEMENTARY_CHARGE_C: float = 1.602176634e-19  # Coulomb
GRAPHENE_BETA: float = 3.0        # beta = -dln(t)/dln(a); literature 2-3.4 (Vozmediano,
                                  # Katsnelson & Guinea, Phys. Rep. 496, 109 (2010))
B_PAIRBREAK_DEFAULT: float = 10.0 # Tesla — SPECULATIVE pseudo-field pair-breaking scale
BUMP_HEIGHT_DEFAULT: float = 5.0  # Angstrom
BUMP_SIGMA_DEFAULT: float = 50.0  # Angstrom
RIPPLE_HEIGHT_DEFAULT: float = 2.0        # Angstrom — intrinsic ripples (Meyer et al.,
RIPPLE_WAVELENGTH_DEFAULT: float = 100.0  # Angstrom    Nature 446, 60 (2007))
BEND_RADIUS_DEFAULT: float = 1000.0  # Angstrom
CAP_RADIUS_DEFAULT: float = 2000.0   # Angstrom

# ---------------------------------------------------------------------------
# Bistritzer-MacDonald / v2 graphene parameters
# ---------------------------------------------------------------------------
W_AA_TBG: float = 0.0797     # eV — AA interlayer tunneling, corrugation-reduced
                             # (Koshino et al., PRX 8, 031087 (2018))
W_AB_TBG: float = 0.0975     # eV — AB interlayer tunneling (same ref)
BM_SHELLS_DEFAULT: int = 3   # momentum-lattice truncation shells
BM_KPOINTS_DEFAULT: int = 16  # k-points per path segment
DOS_BROADENING_MEV: float = 2.0  # Gaussian broadening for DOS
POISSON_GRAPHENE: float = 0.16   # graphene Poisson ratio (Blakslee 1970)
