"""Physical constants and default parameters for moire CPDM calculations."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
HBAR_EV_S: float = 6.582119569e-16          # hbar in eV*s
KB_EV_K: float = 8.617333262e-5             # Boltzmann constant in eV/K

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
BEND_RADIUS_DEFAULT: float = 1000.0  # Angstrom — shared by cylindrical bend and spherical cap

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

# ---------------------------------------------------------------------------
# Pair-density-wave (PDW) modalities (speculative) — Landau theory of
# Zhu et al., PNAS 123, e2602117123 (2026); arXiv:2603.08688.  Field triples
# are (x, y, z) in the paper's vector-magnet frame: x ≈ c*, y = a, z ≈ b*
# (x and z sit ~12° off the c and b axes for the (01-1) cleave).
# ---------------------------------------------------------------------------
TC_UTE2: float = 2.1                # K — molten-salt-flux UTe2 (Supp. Fig. 1)
T_PDW_UTE2: float = 5.0             # K — onset of the q_i CDW, taken as T_PDW
A_TE_UTE2: float = 4.16             # Angstrom — Te-chain repeat; |q_Te| = 2*pi/a
HC2_UTE2: tuple[float, float, float] = (20.0, 11.0, 34.0)
                                    # T — uniform-SC Hc2, approximate; b* > c* > a.
                                    # b*: zero resistance to 34 T near b (Wu et al.,
                                    # PNAS 121, e2403067121 (2024)); a, c*: early-crystal
                                    # 6.5 T and 11 T (Aoki et al., JPSJ 89, 053705
                                    # (2020)) scaled by (Tc ratio)^2 ~ (2.1/1.6)^2
H_PDW_UTE2: tuple[float, float, float] = (14.0, 3.75, 37.5)
                                    # T — PDW critical fields at T = 0, chosen so the
                                    # linear GL law puts H_sup(4.2 K) at the middle of
                                    # the measured windows: c* 1.5-3, a 0.4-0.8,
                                    # b* 4-8 T (Zhu et al. Fig. 4)
H_ROT_UTE2: tuple[float, float, float] = (0.4, 0.4, 4.5)
                                    # T — d-vector rotation onset, just below the
                                    # first fields where p2 is seen at 300 mK:
                                    # ~0.5 T (c*, a) and 5 T (b*) (Fig. 3)
OVERLAP0_UTE2: tuple[float, float, float] = (0.0, 0.0, 1.0)
                                    # d0·d_{p_i} at B = 0: p1, p2 orthogonal (absent
                                    # at zero field), p3 aligned (seen at zero field)
P_VECTORS_UTE2_QTE: tuple[tuple[float, float], ...] = (
    (-0.13, -0.35),
    (0.13, -0.35),
    (0.0, -0.28),
)                                   # PDW wavevectors p_i in units of |q_Te|, from
                                    # p3 = (0, -0.28) and p_i = p3 - h_i/2 with the
                                    # mirror-consistent h2 = (0.26, +0.14) (the paper
                                    # prints -0.14, which breaks h1 + h2 + p3 = 0)
PDW_RATIO_UTE2: float = 0.3         # Delta_p(0) / Delta_0(0) — illustrative only
DELTA0_UTE2: float = BCS_GAP_RATIO * KB_EV_K * 1.0e3 * TC_UTE2
                                    # meV — weak-coupling BCS gap for Tc = 2.1 K
PDW_KAPPA_P: float = 1.0            # g1*Delta0*Delta_p/r1 — p-CDW (PDW x uniform SC);
                                    # with the floor below, p2 shows at 5-6 T but not
                                    # 8 T along b* at 1.6 K (Supp. Fig. 8)
PDW_KAPPA_Q: float = 1.0            # g2*Delta_p^2/r2 — q = 2p CDW (PDW alone)
PDW_KAPPA_H: float = 1.0            # gamma2/r4 — h = 2p3 - 2p_j harmonics
PDW_KAPPA_HARM: float = 0.0         # gamma1/r2 — q fed as a harmonic of p; 0 is the
                                    # paper's reading (onset temperatures differ)
PDW_VISIBILITY_FLOOR: float = 0.01  # |rho_k|^2 below this counts as FFT background
TC_FETE_FILM: float = 13.5          # K — stoichiometric FeTe films (Yan et al.,
                                    # Nature 652, 342 (2026)); moire-locked preset

# ---------------------------------------------------------------------------
# Spin-fluctuation pairing (speculative module) — LCCO magnetotransport of
# Duffy et al., Nat. Phys. (2026), doi:10.1038/s41567-026-03454-6;
# arXiv:2502.13612 (page numbers: arXiv v1 PDF).  Rates in ps^-1, k in 1/m,
# v in m/s, fields in T, lattice lengths in Angstrom.
# ---------------------------------------------------------------------------
KB_J_K: float = 1.380649e-23            # J/K, exact SI (KB_EV_K above is eV/K)
A_LCCO: float = 4.01                    # Angstrom — Methods, p. 13
C_LCCO: float = 12.38                   # Angstrom — Methods, p. 13
LAYER_SPACING_LCCO: float = C_LCCO / 2.0  # Angstrom — one CuO2 plane per c/2 (inferred:
                                        # 0.841-hole Drude R_H, Fig. 4b2 rho, Yuan 2022 A1/d)
KF0_LCCO: float = 5.424e9               # 1/m — Eq. 7, p. 18
KF_ANISOTROPY_LCCO: float = 0.11        # alpha in Eq. 7
VF0_LCCO: float = 4.946e5               # m/s — Eq. 8
VF_ANISOTROPY_LCCO: float = 0.03        # beta in Eq. 8
GAMMA_IMP_LCCO: float = 27.0            # ps^-1 — Table 2 (x = 0.159, film S1), p. 15
ALPHA1_LCCO: float = 0.75               # ps^-1 K^-1 — Table 2 (times g(x))
ALPHA2_LCCO: float = 0.0112             # ps^-1 K^-2 — Table 2
HOTSPOT_PEAK_LCCO: float = 31.4         # ps^-1 — effective height plotted in Fig. 4b1 and used in
                                        # its simulations (inferred; Table 2 prints 50)
HOTSPOT_PEAK_TABLE2: float = 50.0       # ps^-1 — Table 2 as printed; does not reproduce Fig. 4
HOTSPOT_WIDTH_LCCO: float = 0.05        # rad — Table 2 "HS width", read as the 1/e half-width w
                                        # of exp(-(dphi/w)^2) (inferred from Fig. 4b1)
HOTSPOT_PEAK_FIG9C: float = 105.0       # ps^-1 — Fig. 9c, p. 14
PHI_HS_DEG_LCCO: float = 27.5           # deg — Fig. 4 caption (the FS meets the AFM zone
                                        # boundary at 27.20 deg)
X_FSR_LCCO: float = 0.14                # end of the Fermi-surface reconstruction, p. 3
SF_DOPING_REF: float = 0.159            # g = 1 here (Tables 3-4)
G_TABLE_LCCO: tuple[tuple[float, float], ...] = (
    (0.116, 6.50), (0.119, 5.75), (0.120, 5.20), (0.125, 3.85), (0.131, 2.90), (0.134, 2.52),
    (0.138, 2.10), (0.141, 1.87), (0.146, 1.46), (0.149, 1.45), (0.153, 1.38), (0.156, 1.18),
    (0.159, 1.00), (0.165, 0.85), (0.169, 0.76), (0.174, 0.66), (0.180, 0.57), (0.184, 0.55),
    (0.188, 0.55),
)                                       # Table 3 (S1, MR fits at 40 K / 20 T) + Table 4
                                        # (S2, from A1)
TC_LABELS_FIG5C: tuple[tuple[float, float], ...] = (
    (0.12, 25.0), (0.14, 15.0), (0.16, 8.0), (0.18, 0.0),
)                                       # K — Fig. 5c top-axis labels on the bottom ticks
                                        # (vector-located)
TC_TABLE_LCCO: tuple[tuple[float, float], ...] = (
    (0.100, 25.0), (0.120, 25.0), (0.140, 15.0), (0.160, 8.0), (0.180, 0.0), (0.190, 0.0),
)                                       # K — piecewise linear; plateau below 0.12 inferred
                                        # (Jin 2011, Yuan 2022)
SF_NU: int = 12                         # sin^nu exponent, p. 6 (from Grissonnanche et al. 2021)
SF_CVC_EPSILON: float = 1.0             # peak of eps(phi), Eq. 13 — Fig. 12a needs ~1;
                                        # 1/(1-eps^2) dropped (p. 19)
SF_CVC_SIGMA: float = 0.07              # rad — Eq. 13, p. 19
SF_FIG13_RH: tuple[float, float, float, float] = (8.7, 5.6, 10.3, 4.3)
                                        # 1e-10 m^3/C, Fig. 13 (a)-(d), T not stated (p. 21);
                                        # (d)/(a) = 0.494
SF_N_PHI: int = 1024                    # Fermi-surface samples (power of two)
SF_FIELD_STEP: float = 0.5              # T
SF_FIELD_MAX: float = 35.0              # T — HFML 35 T Bitter magnet (p. 10)
SF_MR_TEMPERATURES: tuple[float, ...] = (20.0, 30.0, 40.0, 50.0)   # K — Fig. 4
SF_FIT_TEMPERATURE: float = 40.0        # K — g(x) fitted at 40 K and 20 T (Table 3)
SF_FIT_FIELD: float = 20.0              # T
SF_SLOPE_FIELD: float = 33.0            # T — gamma_1 = drho/dmu0H at 33 T (Fig. 15)
SF_RHO_FIT_RANGE: tuple[float, float] = (15.0, 60.0)  # K — Fig. 6 caption
# Part B (SPECULATIVE): pairing maps calibrated to the Fig. 5c Tc labels
SF_POWER_LAW_LAMBDA_C: float = 0.30462015776911294
SF_POWER_LAW_EXPONENT: float = 0.3926512381199364
SF_POWER_LAW_LAMBDA_REF: float = 2.1714302476009117   # lambda_tr at x = 0.12
SF_POWER_LAW_TC_REF: float = 25.0                      # K
SF_MCMILLAN_OMEGA_K: float = 178.04201155099287        # K — best fit; mu* pinned at 0 (fails)
SF_MCMILLAN_MU_STAR: float = 0.0
SF_GLUE_ETA: float = 0.10               # depth of the what-if glue modulation (illustrative)
SF_GLUE_COHERENCE: str = "clean"        # default coherence model: "clean" | "dirty" | "local"
# Part D: hole-doped quadrature contrast, Tl2201 (Ayres et al., Nature 595, 661 (2021), ED Fig. 2)
SF_QUAD_RHO0: float = 15.5              # micro-Ohm cm
SF_QUAD_A: float = 0.14                 # micro-Ohm cm / K
SF_QUAD_B: float = 0.003                # micro-Ohm cm / K^2
SF_QUAD_ALPHA_KB: float = 0.04          # micro-Ohm cm / K
SF_QUAD_GAMMA_MUB: float = 0.20         # micro-Ohm cm / T
