"""Topological surface state physics and 3D Cooper surface construction.

Implements proximity effect z-decay, 3D gap field extension, Dirac surface
state dispersion, and speculative features (Majorana zero modes, topological
phase boundaries, magnetoelectric response, Chern number estimates).

Established physics unless marked SPECULATIVE.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import j0 as bessel_j0

from waytogocoop.config import (
    ANGSTROM_TO_M,
    G_FACTOR_TSS,
    HBAR_EV_S,
    K_F_TSS,
    MU_B_EV_T,
    V_F_TI,
    XI_MAJORANA_DEFAULT,
    XI_PROXIMITY_DEFAULT,
    ZERO_THRESHOLD,
)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ProximityConfig:
    """Configuration for proximity effect z-decay."""

    xi_prox: float = XI_PROXIMITY_DEFAULT          # Proximity coherence length (Angstrom)
    n_z_layers: int = 50                            # Number of z-slices
    z_min: float = -50.0                            # Into superconductor (Angstrom)
    z_max: float = 300.0                            # Into topological insulator (Angstrom)
    interface_transparency: float = 0.8             # BTK interface transmission T ∈ (0, 1]


@dataclass
class ProximityResult:
    """3D gap field with z-decay."""

    gap_3d: np.ndarray              # Shape (n_z, grid_y, grid_x) — meV
    z_coords: np.ndarray            # 1D z positions in Angstrom
    decay_profile: np.ndarray       # 1D f(z) profile
    resolution: int                 # xy grid size
    n_z: int                        # number of z layers


@dataclass
class MajoranaResult:
    """SPECULATIVE: Majorana zero mode probability density at vortex cores."""

    probability_density: np.ndarray   # 2D array
    localization_length: float        # xi_M (Angstrom)
    n_vortices_with_mzm: int          # Count of vortices hosting MZMs


@dataclass
class CooperSurface3D:
    """Full 3D Cooper-pair order parameter surface."""

    field_3d: np.ndarray          # (n_z, grid_y, grid_x)
    x: np.ndarray                 # 1D x coordinates (Angstrom)
    y: np.ndarray                 # 1D y coordinates (Angstrom)
    z: np.ndarray                 # 1D z coordinates (Angstrom)
    moire_period: float           # Angstrom
    vortex_period: float          # Angstrom (inf if no field)


# ---------------------------------------------------------------------------
# Established physics — proximity effect
# ---------------------------------------------------------------------------

def proximity_decay_profile(
    z: np.ndarray,
    xi_prox: float = XI_PROXIMITY_DEFAULT,
    interface_transparency: float = 0.8,
) -> np.ndarray:
    """Proximity-induced gap decay profile f(z).

    z < 0  (inside SC):   f(z) = 1.0
    z = 0  (interface):   f(z) = T   (interface transparency)
    z > 0  (inside TI):   f(z) = T * exp(−z / xi_prox)

    Established physics: proximity effect with BTK interface model.
    """
    if xi_prox <= 0:
        raise ValueError("xi_prox must be positive")
    if not (0 < interface_transparency <= 1):
        raise ValueError("interface_transparency must be in (0, 1]")
    z = np.asarray(z, dtype=np.float64)
    profile = np.ones_like(z)
    ti_mask = z >= 0
    profile[ti_mask] = interface_transparency * np.exp(-z[ti_mask] / xi_prox)
    return profile


def gap_3d(
    gap_2d: np.ndarray,
    config: ProximityConfig | None = None,
) -> ProximityResult:
    """Extend a 2D gap field into 3D by applying proximity z-decay.

    Delta(x,y,z) = Delta(x,y) * f(z)

    Returns ProximityResult with the full 3D gap array.
    """
    if config is None:
        config = ProximityConfig()

    z_coords = np.linspace(config.z_min, config.z_max, config.n_z_layers)
    profile = proximity_decay_profile(z_coords, config.xi_prox, config.interface_transparency)

    # gap_2d is (ny, nx); result is (n_z, ny, nx)
    gap_3d_array = profile[:, np.newaxis, np.newaxis] * gap_2d[np.newaxis, :, :]

    return ProximityResult(
        gap_3d=gap_3d_array,
        z_coords=z_coords,
        decay_profile=profile,
        resolution=gap_2d.shape[0],
        n_z=config.n_z_layers,
    )


# ---------------------------------------------------------------------------
# Established physics — Dirac surface state
# ---------------------------------------------------------------------------

def dirac_dispersion(
    k: np.ndarray,
    v_F: float = V_F_TI,
) -> np.ndarray:
    """Topological surface state Dirac cone dispersion.

    E(k) = hbar * v_F * |k|

    k in 1/Angstrom, returns energy in meV.
    """
    if v_F <= 0:
        raise ValueError("v_F must be positive")
    k = np.asarray(k, dtype=np.float64)
    # hbar in eV·s, v_F in m/s, k in 1/Angstrom = 1/ANGSTROM_TO_M /m
    E_eV = HBAR_EV_S * v_F * np.abs(k) * (1.0 / ANGSTROM_TO_M)
    return E_eV * 1.0e3  # eV → meV


# ---------------------------------------------------------------------------
# 3D Cooper surface construction
# ---------------------------------------------------------------------------

def cooper_surface_3d(
    gap_2d: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    moire_period: float,
    vortex_period: float = float("inf"),
    suppression_field: np.ndarray | None = None,
    proximity_config: ProximityConfig | None = None,
) -> CooperSurface3D:
    """Construct the full 3D Cooper-pair order parameter surface.

    Combines:
      1. Moire-modulated 2D gap Delta(x,y)
      2. Vortex suppression (if suppression_field provided)
      3. Proximity z-decay

    Returns CooperSurface3D with the volumetric field.
    """
    if proximity_config is None:
        proximity_config = ProximityConfig()

    # Apply vortex suppression if present
    effective_gap = gap_2d.copy()
    if suppression_field is not None:
        effective_gap *= suppression_field

    # Extend to 3D
    result = gap_3d(effective_gap, proximity_config)

    return CooperSurface3D(
        field_3d=result.gap_3d,
        x=np.asarray(x),
        y=np.asarray(y),
        z=result.z_coords,
        moire_period=moire_period,
        vortex_period=vortex_period,
    )


# ---------------------------------------------------------------------------
# SPECULATIVE — Majorana zero modes
# ---------------------------------------------------------------------------

def majorana_probability_density(
    x: np.ndarray,
    y: np.ndarray,
    vortex_positions: np.ndarray,
    xi_M: float = XI_MAJORANA_DEFAULT,
    k_F: float = K_F_TSS,
) -> MajoranaResult:
    """SPECULATIVE: Majorana zero mode probability density.

    |psi_MZM(r)|^2 ~ sum_v  exp(−2|r−r_v|/xi_M) * J_0(k_F * |r−r_v|)^2

    Based on: Fu & Kane, PRL 100, 096407 (2008).

    The exact wavefunction profile depends on microscopic details
    (vortex core structure, spin-orbit coupling strength) not captured here.
    This is a qualitative envelope model.

    Returns MajoranaResult with 2D probability density.
    """
    if xi_M <= 0:
        raise ValueError("xi_M must be positive")
    if k_F <= 0:
        raise ValueError("k_F must be positive")
    X, Y = np.meshgrid(x, y)
    density = np.zeros_like(X)

    if vortex_positions.size == 0:
        return MajoranaResult(
            probability_density=density,
            localization_length=xi_M,
            n_vortices_with_mzm=0,
        )

    for vx, vy in vortex_positions:
        dx = X - vx
        dy = Y - vy
        r = np.hypot(dx, dy)
        envelope = np.exp(-2.0 * r / xi_M)
        oscillation = bessel_j0(k_F * r) ** 2
        density += envelope * oscillation

    # Normalise to peak = 1
    peak = density.max()
    if peak > 1.0e-30:
        density /= peak

    return MajoranaResult(
        probability_density=density,
        localization_length=xi_M,
        n_vortices_with_mzm=len(vortex_positions),
    )


def majorana_probability_density_3d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    vortex_positions: np.ndarray,
    xi_M: float = XI_MAJORANA_DEFAULT,
    k_F: float = K_F_TSS,
    xi_prox: float = XI_PROXIMITY_DEFAULT,
) -> np.ndarray:
    """SPECULATIVE: 3D Majorana zero mode probability density.

    Extends :func:`majorana_probability_density` (the 2D in-plane envelope
    × Bessel oscillation) into z by multiplying with an exponential decay
    ``exp(-2|z|/xi_prox)`` anchored at the interface (z = 0). The factor of 2
    matches the probability density (|ψ|² inherits twice the wavefunction
    decay rate).

    Returns an array of shape ``(nz, ny, nx)`` normalised so that the peak
    across the full volume is 1. If there are no vortices, the volume is
    all zeros.
    """
    if xi_prox <= 0:
        raise ValueError("xi_prox must be positive")
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    planar = majorana_probability_density(x, y, vortex_positions, xi_M=xi_M, k_F=k_F)
    envelope_z = np.exp(-2.0 * np.abs(z) / xi_prox)  # (nz,)
    density_3d = envelope_z[:, None, None] * planar.probability_density[None, :, :]

    peak = density_3d.max()
    if peak > 1.0e-30:
        density_3d = density_3d / peak
    return density_3d


# ---------------------------------------------------------------------------
# SPECULATIVE — topological phase boundary
# ---------------------------------------------------------------------------

def topological_phase_index(
    zeeman_meV: float,
    delta_meV: float,
    mu_meV: float = 0.0,
) -> int:
    """SPECULATIVE: Topological phase index.

    Simplified Fu-Kane criterion:
      Topological (index 1) when  E_Z > sqrt(Delta^2 + mu^2)
      Trivial     (index 0) otherwise

    mu is the chemical potential offset from the Dirac point.
    """
    if delta_meV < 0:
        raise ValueError("delta_meV must be >= 0")
    threshold = np.hypot(delta_meV, mu_meV)
    return 1 if zeeman_meV > threshold else 0


def phase_diagram_sweep(
    B_values: np.ndarray,
    delta_values: np.ndarray,
    g_factor: float = G_FACTOR_TSS,
    mu_meV: float = 0.0,
) -> np.ndarray:
    """SPECULATIVE: Compute topological phase diagram over (B, Delta) space.

    Returns 2D array of phase indices:  shape (len(delta_values), len(B_values)).
    """
    if g_factor <= 0:
        raise ValueError("g_factor must be positive")
    phase = np.zeros((len(delta_values), len(B_values)), dtype=np.int32)
    for i, delta in enumerate(delta_values):
        for j, B in enumerate(B_values):
            E_Z = g_factor * MU_B_EV_T * B * 1.0e3  # eV → meV
            phase[i, j] = topological_phase_index(E_Z, delta, mu_meV)
    return phase


# ---------------------------------------------------------------------------
# SPECULATIVE — topological magnetoelectric effect
# ---------------------------------------------------------------------------

E_CHARGE: float = 1.602176634e-19  # Coulombs
H_PLANCK: float = 6.62607015e-34   # J·s

def topological_magnetoelectric_polarization(
    B_tesla: float,
    theta: float = np.pi,
) -> float:
    """SPECULATIVE: Topological magnetoelectric effect surface polarization.

    P = (e^2 / 2 pi h) * theta * B

    theta = pi for a time-reversal-invariant topological insulator.
    Returns polarization in C/m^2.

    Observable via anomalous Hall conductivity in magnetotransport.
    """
    return (E_CHARGE**2 / (2.0 * np.pi * H_PLANCK)) * theta * B_tesla


def chern_number_estimate(
    delta_meV: float,
    zeeman_meV: float,
    mu_meV: float = 0.0,
) -> float:
    """SPECULATIVE: Chern number estimate for proximitized TI surface.

    C = sign(E_Z^2 − Delta^2 − mu^2) / 2

    Returns ±0.5 or 0.  Observable as quantized Hall plateau.
    """
    discriminant = zeeman_meV**2 - delta_meV**2 - mu_meV**2
    if abs(discriminant) < ZERO_THRESHOLD:
        return 0.0
    return 0.5 * np.sign(discriminant)
