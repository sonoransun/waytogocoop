"""Magnetic field effects on moire-modulated Cooper-pair density.

Implements Abrikosov vortex lattice generation, Ginzburg-Landau vortex core
profiles, Zeeman splitting, flux quantization, Meissner screening currents,
and speculative tertiary effects (field-tunable CPDM, moire-vortex beating,
local susceptibility).

Established physics unless marked SPECULATIVE.

CAVEATS for non-FeTe/TI substrates
----------------------------------
Several default constants in this module are calibrated for the
FeTe/topological-insulator heterostructure the project was originally built
around:

- ``G_FACTOR_TSS`` ≈ 30 (topological surface state g-factor)
- ``LAMBDA_L_FETE`` = 150 Å  (London penetration depth of FeTe)
- ``DEFAULT_COHERENCE_LENGTH`` = 20 Å  (FeTe BCS coherence length)

For twisted bilayer graphene and other non-TI substrates these are not
physically appropriate — use the UI sliders to override the g-factor
(graphene is near the free-electron value g ≈ 2), the coherence length, and
the proximity parameters. The vortex lattice geometry, flux quantization,
and Zeeman energy formulas themselves are material-agnostic and remain
valid once the constants are retuned.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from waytogocoop.computation.superconducting import cpdm_amplitude
from waytogocoop.config import (
    ANGSTROM_TO_M,
    BC2_FETE,
    DEFAULT_COHERENCE_LENGTH,
    G_FACTOR_TSS,
    LAMBDA_L_FETE,
    MU_B_MEV_T,
    PHI_0,
    ZERO_THRESHOLD,
)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MagneticFieldConfig:
    """Configuration for applied magnetic field."""

    Bx: float = 0.0  # In-plane field component (Tesla)
    By: float = 0.0  # In-plane field component (Tesla)
    Bz: float = 0.0  # Perpendicular field component (Tesla)


@dataclass
class VortexLatticeResult:
    """Result of Abrikosov vortex lattice computation."""

    vortex_positions: np.ndarray  # (N_v, 2) in Angstrom
    vortex_period: float          # Angstrom
    flux_per_moire_cell: float    # Phi / Phi_0
    is_commensurate: bool         # Whether a_v ≈ L_m within 5 %
    commensuration_field: float   # Tesla
    suppression_field: np.ndarray  # 2D, values in [0, 1]


@dataclass
class ZeemanResult:
    """Result of in-plane Zeeman effect computation."""

    zeeman_energy: float        # meV
    pauli_limit_field: float    # Tesla
    depairing_ratio: float      # |B_parallel| / B_P


@dataclass
class MagnetometricResponse:
    """Local magnetometric response fields."""

    susceptibility_map: np.ndarray  # 2D chi(x,y)
    screening_jx: np.ndarray        # 2D Jx(x,y)
    screening_jy: np.ndarray        # 2D Jy(x,y)


# ---------------------------------------------------------------------------
# Established physics — Abrikosov vortex lattice
# ---------------------------------------------------------------------------

def vortex_lattice_period(Bz: float) -> float:
    """Triangular Abrikosov vortex lattice period.

    a_v = sqrt(2 * Phi_0 / (sqrt(3) * |Bz|))

    Abrikosov (1957).  Returns period in Angstrom.
    Returns inf for Bz = 0.
    """
    Bz_abs = abs(Bz)
    if Bz_abs < 1.0e-15:
        return float("inf")
    a_v_m = np.sqrt(2.0 * PHI_0 / (np.sqrt(3.0) * Bz_abs))
    return a_v_m / ANGSTROM_TO_M  # convert m → Angstrom


def commensuration_field(moire_period_angstrom: float) -> float:
    """Field at which vortex lattice period equals moire period.

    B_comm = 2 * Phi_0 / (sqrt(3) * L_m^2)

    Returns Tesla.  Returns inf if moire period is zero or infinite.
    """
    L_m = moire_period_angstrom * ANGSTROM_TO_M  # Angstrom → metre
    if L_m < 1.0e-30 or not np.isfinite(moire_period_angstrom):
        return float("inf")
    return 2.0 * PHI_0 / (np.sqrt(3.0) * L_m**2)


def generate_vortex_positions(
    Bz: float,
    physical_extent: float,
    grid_size: int,
) -> np.ndarray:
    """Generate triangular lattice of vortex-core positions.

    Fills the viewport [−extent/2, extent/2]^2 with vortex positions.
    Returns shape (N_v, 2) in Angstrom.  Empty array for Bz ≈ 0.
    """
    a_v = vortex_lattice_period(Bz)
    if not np.isfinite(a_v) or a_v <= 0:
        return np.empty((0, 2), dtype=np.float64)

    half = physical_extent / 2.0
    # Triangular lattice: a1 = (a_v, 0), a2 = (a_v/2, a_v*sqrt(3)/2)
    a2y = a_v * np.sqrt(3.0) / 2.0

    # Number of rows/cols needed
    n_cols = int(np.ceil(physical_extent / a_v)) + 2
    n_rows = int(np.ceil(physical_extent / a2y)) + 2

    positions = []
    for row in range(-n_rows, n_rows + 1):
        y = row * a2y
        x_offset = (row % 2) * a_v / 2.0
        for col in range(-n_cols, n_cols + 1):
            x = col * a_v + x_offset
            if -half <= x <= half and -half <= y <= half:
                positions.append([x, y])

    if not positions:
        return np.empty((0, 2), dtype=np.float64)
    return np.array(positions, dtype=np.float64)


def vortex_suppression_field(
    x: np.ndarray,
    y: np.ndarray,
    vortex_positions: np.ndarray,
    coherence_length: float = DEFAULT_COHERENCE_LENGTH,
) -> np.ndarray:
    """Order parameter suppression from vortex cores.

    suppression(r) = Product_v  tanh(|r − r_v| / xi)

    Ginzburg-Landau vortex profile.  Returns 2D array in [0, 1].
    Vortices farther than 5*xi are approximated as tanh→1.
    """
    if coherence_length <= 0:
        raise ValueError("coherence_length must be positive")
    X, Y = np.meshgrid(x, y)
    suppression = np.ones_like(X)

    if vortex_positions.size == 0:
        return suppression

    cutoff = 5.0 * coherence_length
    for vx, vy in vortex_positions:
        dx = X - vx
        dy = Y - vy
        dist = np.hypot(dx, dy)
        mask = dist < cutoff
        if np.any(mask):
            factor = np.ones_like(dist)
            factor[mask] = np.tanh(dist[mask] / coherence_length)
            suppression *= factor

    return suppression


def combined_gap_with_vortices(
    gap_2d: np.ndarray,
    suppression: np.ndarray,
) -> np.ndarray:
    """Combine moire gap modulation with vortex suppression.

    Delta_total(x,y) = Delta_moire(x,y) * suppression(x,y)
    """
    return gap_2d * suppression


# ---------------------------------------------------------------------------
# Established physics — flux quantization and Zeeman
# ---------------------------------------------------------------------------

def flux_per_moire_cell(Bz: float, moire_period_angstrom: float) -> float:
    """Quantized magnetic flux through one moire unit cell.

    n = |Bz| * A_moire / Phi_0

    where A_moire = (sqrt(3)/2) * L_m^2  for a hexagonal moire cell.
    Returns dimensionless ratio Phi/Phi_0.
    """
    L_m = moire_period_angstrom * ANGSTROM_TO_M
    A_moire = (np.sqrt(3.0) / 2.0) * L_m**2
    return abs(Bz) * A_moire / PHI_0


def zeeman_energy(
    Bx: float,
    By: float,
    g_factor: float = G_FACTOR_TSS,
) -> float:
    """Zeeman splitting energy from in-plane magnetic field.

    E_Z = g * mu_B * |B_parallel|

    Returns meV.
    """
    B_parallel = np.hypot(Bx, By)
    return g_factor * MU_B_MEV_T * B_parallel


def pauli_limiting_field(delta_avg_meV: float) -> float:
    """Pauli paramagnetic limiting field for s-wave pairing.

    B_P = Delta / (sqrt(2) * mu_B)

    Returns Tesla.
    """
    if delta_avg_meV <= 0:
        return float("inf")
    return delta_avg_meV / (np.sqrt(2.0) * MU_B_MEV_T)


def compute_zeeman(config: MagneticFieldConfig, delta_avg_meV: float) -> ZeemanResult:
    """Compute full Zeeman result from field config."""
    E_Z = zeeman_energy(config.Bx, config.By)
    B_P = pauli_limiting_field(delta_avg_meV)
    B_par = np.hypot(config.Bx, config.By)
    ratio = B_par / B_P if np.isfinite(B_P) and B_P > 0 else 0.0
    return ZeemanResult(
        zeeman_energy=E_Z,
        pauli_limit_field=B_P,
        depairing_ratio=ratio,
    )


# ---------------------------------------------------------------------------
# Established physics — Meissner screening currents
# ---------------------------------------------------------------------------

def screening_currents(
    x: np.ndarray,
    y: np.ndarray,
    vortex_positions: np.ndarray,
    lambda_L: float = LAMBDA_L_FETE,
) -> tuple[np.ndarray, np.ndarray]:
    """Meissner screening supercurrent density around vortex cores.

    Each vortex carries one flux quantum.  The azimuthal current around
    a single vortex at distance r is:

        J_theta(r) ~ (Phi_0 / (2 pi mu_0 lambda_L^2 r)) * exp(-r / lambda_L)

    Returns (Jx, Jy) 2D arrays in arbitrary units, normalised to peak = 1.
    """
    if lambda_L <= 0:
        raise ValueError("lambda_L must be positive")
    X, Y = np.meshgrid(x, y)
    Jx = np.zeros_like(X)
    Jy = np.zeros_like(Y)

    if vortex_positions.size == 0:
        return Jx, Jy

    for vx, vy in vortex_positions:
        dx = X - vx
        dy = Y - vy
        r = np.hypot(dx, dy)
        r_safe = np.maximum(r, ZERO_THRESHOLD)

        magnitude = np.where(r < ZERO_THRESHOLD, 0.0, np.exp(-r / lambda_L) / r_safe)
        # Azimuthal direction: (-dy, dx) / r
        Jx += -dy / r_safe * magnitude
        Jy += dx / r_safe * magnitude

    # Normalise to peak = 1
    J_mag = np.sqrt(Jx**2 + Jy**2)
    peak = J_mag.max()
    if peak > ZERO_THRESHOLD:
        Jx /= peak
        Jy /= peak

    return Jx, Jy


# ---------------------------------------------------------------------------
# SPECULATIVE — local susceptibility
# ---------------------------------------------------------------------------

def local_susceptibility(
    gap_field: np.ndarray,
    lambda_L: float = LAMBDA_L_FETE,
) -> np.ndarray:
    """SPECULATIVE: Local magnetic susceptibility chi(r).

    Model:  chi(r) ~ -(1/lambda_L^2) * [1 − (Delta(r)/Delta_max)^2]

    Qualitative only — actual susceptibility requires self-consistent
    Bogoliubov–de Gennes calculation.

    Returns 2D array normalised to [−1, 0].
    """
    gap_max = np.max(np.abs(gap_field))
    if gap_max < ZERO_THRESHOLD:
        return np.zeros_like(gap_field)
    normalised = gap_field / gap_max
    chi = -(1.0 - normalised**2)
    return chi


# ---------------------------------------------------------------------------
# SPECULATIVE — tertiary physics
# ---------------------------------------------------------------------------

def moire_vortex_beating(
    moire_period: float,
    vortex_period: float,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """SPECULATIVE: Beating pattern from moire and vortex lattice interference.

    When the two lattices are incommensurate, a secondary superlattice
    emerges with period:  L_beat = L_m * L_v / |L_m − L_v|

    Observable via STM modulation at the beating period.

    Returns 2D intensity pattern normalised to [0, 1].
    """
    if moire_period <= 0 or not np.isfinite(moire_period):
        raise ValueError("moire_period must be positive and finite")
    if vortex_period <= 0:
        raise ValueError("vortex_period must be positive")
    if abs(moire_period - vortex_period) < ZERO_THRESHOLD:
        return np.ones((len(y), len(x)), dtype=np.float64)

    L_beat = moire_period * vortex_period / abs(moire_period - vortex_period)
    q_beat = 2.0 * np.pi / L_beat

    X, Y = np.meshgrid(x, y)
    # Hexagonal beating: 6-fold symmetric
    pattern = np.zeros_like(X)
    for i in range(6):
        angle = i * np.pi / 3.0
        gx = q_beat * np.cos(angle)
        gy = q_beat * np.sin(angle)
        pattern += np.cos(gx * X + gy * Y)

    # Normalise to [0, 1]
    pmin, pmax = pattern.min(), pattern.max()
    rng = pmax - pmin
    if rng > 1.0e-15:
        pattern = (pattern - pmin) / rng
    else:
        pattern = np.full_like(pattern, 0.5)

    return pattern


def field_tunable_cpdm(
    moire_period: float,
    coherence_length: float,
    Bz: float,
    Bc2: float = BC2_FETE,
) -> float:
    """SPECULATIVE: CPDM amplitude modulated by applied field.

    A_CPDM(B) = A_CPDM(0) * (1 − |Bz|/Bc2)

    Speculative ansatz — no experimental validation for moire systems.
    The vortex cores suppress the gap, effectively reducing the modulation
    contrast.  Returns dimensionless amplitude ∈ [0, 1].
    """
    if coherence_length <= 0:
        raise ValueError("coherence_length must be positive")
    if Bc2 <= 0:
        raise ValueError("Bc2 must be positive")
    A0 = cpdm_amplitude(moire_period, coherence_length)
    field_ratio = min(abs(Bz) / Bc2, 1.0)

    return A0 * (1.0 - field_ratio)


def commensuration_pinning_energy(
    moire_period: float,
    vortex_period: float,
) -> float:
    """SPECULATIVE: Cosine pinning energy for vortex-moire commensuration.

    E_pin ~ cos(2 pi L_m / L_v)

    Peaks at integer ratios L_m/L_v indicating enhanced vortex pinning.
    Observable as anomaly in critical current vs. field.

    Returns dimensionless energy in [−1, 1].
    """
    if vortex_period < ZERO_THRESHOLD or not np.isfinite(vortex_period):
        return 0.0
    return float(np.cos(2.0 * np.pi * moire_period / vortex_period))
