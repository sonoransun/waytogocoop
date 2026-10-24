"""Pair-density-wave (PDW) modalities and the charge orders they leave behind.

SPECULATIVE throughout — a phenomenological Landau model, not a microscopic
theory.  It implements the free energy Zhu et al. use to explain the field-
and temperature-sensitive charge density waves (CDWs) that vector-field STM
resolves on the spin-triplet superconductor UTe2 (PNAS 123, e2602117123
(2026); arXiv:2603.08688):

    F = sum_i [a_i (T - T_PDW) |d_{p_i}|^2 + l_i |d_{p_i}|^4]
        + a_0 (T - Tc) |d_0|^2 + l_0 |d_0|^4
        + r1 sum_i |rho_{p_i}|^2 + r2 sum_i |rho_{2p_i}|^2
        + r4 (|rho_{2p3-2p2}|^2 + |rho_{2p3-2p1}|^2)
        + g1 sum_i rho_{p_i} [d_0 . d*_{p_i} + d*_0 . d_{-p_i}]
        + g2 sum_i rho_{2p_i} d_{-p_i} . d*_{p_i}
        + gamma1 sum_i rho_{p_i}^2 rho*_{2p_i}
        + gamma2 [rho_{2p3} rho*_{2p2} rho*_{2p3-2p2} + (2 -> 1)] + c.c.

d_0 is the d-vector of the uniform triplet condensate and d_{+-p_i} those of
the PDW at wavevectors +-p_i (i = 1, 2, 3).  Minimising over the CDW
amplitudes gives three families of composite charge order (Table 1):

- p_i — rho_{p_i} ~ d*_0 . d_{p_i} + d_0 . d*_{-p_i}: PDW times uniform SC,
  so it exists only below Tc.
- q_i = 2 p_i — rho_{2p_i} ~ d*_{-p_i} . d_{p_i}: the PDW alone, so it
  survives between Tc and T_PDW.  This is the "Cheshire cat's grin" regime:
  pairing without uniform superconductivity, visible only as charge order.
- h_1 = 2p_3 - 2p_2, h_2 = 2p_3 - 2p_1 — rho_h ~ rho_{q3} rho*_{q_j}: fourth
  order in the PDW, so it fades first on warming.

Model choices beyond the paper (also SPECULATIVE):

- Mean-field amplitudes with a linear Ginzburg-Landau field term,
  |d|^2 / |d(0)|^2 = max(0, 1 - T/T* - h), where h = sqrt(sum_n (B_n/H_n)^2)
  is an ellipsoidal reduced field, so H*(T) = H*(0) (1 - T/T*) on each axis.
- Field-induced d-vector rotation (the paper's account of the field-induced
  p1, p2): the part of d_{p_i} orthogonal to d_0 shrinks as H_rot/H above
  the onset, sin(theta) = sin(theta_0) * min(1, 1/h_rot).
- CDW amplitudes are leading order in the cubic couplings (no back-reaction).
- The composite spin density S_{p_i} ~ |d_0 x d_{p_i}| (the paper's predicted
  surface SDW) uses the rho_{p_i} coupling, so the field converts SDW weight
  into p-CDW weight as the d-vectors align.

Amplitudes are normalised, delta_0 = |d_0(T, B)| / |d_0(0)| and
delta_p = |d_p(T, B)| / |d_p(0)|; the kappa couplings absorb the zero-field
gap scales, so CDW amplitudes are dimensionless.  Field triples are (x, y, z)
in the paper's vector-magnet frame (x ~ c*, y = a, z ~ b* for UTe2).

Mirrors ``crates/moire-core/src/pdw.rs`` — changes here must land in both.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from waytogocoop.computation.fourier import fft_2d
from waytogocoop.config import (
    A_TE_UTE2,
    BC2_FETE,
    DEFAULT_COHERENCE_LENGTH,
    DELTA0_UTE2,
    DELTA_1,
    DELTA_2,
    DELTA_AVG,
    H_PDW_UTE2,
    H_ROT_UTE2,
    HC2_UTE2,
    NORM_FLOOR,
    OVERLAP0_UTE2,
    P_VECTORS_UTE2_QTE,
    PDW_KAPPA_H,
    PDW_KAPPA_HARM,
    PDW_KAPPA_P,
    PDW_KAPPA_Q,
    PDW_RATIO_UTE2,
    PDW_VISIBILITY_FLOOR,
    T_PDW_UTE2,
    TC_FETE_FILM,
    TC_UTE2,
    ZERO_THRESHOLD,
)

PDW_GEOMETRIES: tuple[str, ...] = ("ute2", "moire")
PDW_REGIMES: tuple[str, ...] = ("normal", "pdw_only", "uniform_only", "coexistence")

# Moire-locked directions (deg, before twist): p1/p2 are mirror partners and
# p3 points along -y, matching the UTe2 labelling.
_MOIRE_ANGLES_DEG: tuple[float, float, float] = (30.0, 150.0, 270.0)


def ute2_wavevectors(a_te: float = A_TE_UTE2) -> np.ndarray:
    """PDW wavevectors p_i on the UTe2 surface (1/Angstrom), shape ``(3, 2)``.

    Zhu et al. quote wavevectors in units of the Te Bragg vector
    ``|q_Te| = 2*pi/a_te``; see ``config.P_VECTORS_UTE2_QTE``.
    """
    if a_te <= 0:
        raise ValueError("a_te must be positive")
    return (2.0 * np.pi / a_te) * np.array(P_VECTORS_UTE2_QTE, dtype=float)


def moire_locked_wavevectors(moire_period: float, twist_angle_deg: float = 0.0) -> np.ndarray:
    """Three PDW wavevectors of magnitude 2*pi/L_m at 120 deg (1/Angstrom).

    The three-fold set follows the hexagonal overlayers (every overlayer in
    the database is hexagonal) and rotates with the twist.  It is an
    assumption: on the square FeTe substrate the moire itself is not C3.
    """
    if not np.isfinite(moire_period) or moire_period <= 0:
        raise ValueError("moire_period must be positive and finite")
    angles = np.radians(np.array(_MOIRE_ANGLES_DEG) + twist_angle_deg)
    return (2.0 * np.pi / moire_period) * np.column_stack([np.cos(angles), np.sin(angles)])


def moire_locked_period(
    substrate_a: float, overlayer_a: float, twist_angle_deg: float = 0.0
) -> float:
    """Two-lattice moire period L = a1 a2 / sqrt(a1^2 + a2^2 - 2 a1 a2 cos(theta)).

    Reduces exactly to the mismatch formula a1 a2 / |a1 - a2| at theta = 0
    and to a / (2 sin(theta/2)) for equal lattices, and interpolates
    smoothly between them; ``inf`` for identical lattices at zero twist.
    The moire-locked PDW uses this instead of each app's own period
    estimate so both stacks lock to the same wavevector at any twist.
    """
    if not substrate_a > 0 or not overlayer_a > 0:
        raise ValueError("lattice constants must be positive")
    theta = math.radians(twist_angle_deg)
    d2 = (
        substrate_a * substrate_a
        + overlayer_a * overlayer_a
        - 2.0 * substrate_a * overlayer_a * math.cos(theta)
    )
    if d2 <= ZERO_THRESHOLD:
        return math.inf
    return substrate_a * overlayer_a / math.sqrt(d2)


def moire_locked_pdw_ratio(
    moire_period: float,
    coherence_length: float = DEFAULT_COHERENCE_LENGTH,
    delta_1: float = DELTA_1,
    delta_2: float = DELTA_2,
) -> float:
    """Delta_p(0) / Delta_0(0) of the moire-locked PDW, from the two-gap CPDM.

    Half the gap split (Delta_2 - Delta_1)/2 goes to each of +-Q, relative to
    the mean gap, times the CPDM suppression exp(-xi / L_m).
    """
    if not moire_period > 0 or not coherence_length > 0:
        raise ValueError("moire_period and coherence_length must be positive")
    if not delta_1 + delta_2 > 0:
        raise ValueError("the mean gap must be positive")
    amplitude = (delta_2 - delta_1) / 2.0
    average = (delta_1 + delta_2) / 2.0
    return 0.5 * amplitude / average * math.exp(-coherence_length / moire_period)


def harmonic_vectors(p_vectors) -> np.ndarray:
    """Harmonic wavevectors h1 = 2p3 - 2p2 and h2 = 2p3 - 2p1, shape ``(2, 2)``."""
    p = np.asarray(p_vectors, dtype=float)
    return np.array([2.0 * (p[2] - p[1]), 2.0 * (p[2] - p[0])])


_UTE2_P_VECTORS: tuple[tuple[float, float], ...] = tuple(
    (float(kx), float(ky)) for kx, ky in ute2_wavevectors()
)


@dataclass
class PdwConfig:
    """Parameters of the PDW Landau model; defaults are UTe2 (Zhu et al. 2026)."""

    p_vectors: tuple[tuple[float, float], ...] = _UTE2_P_VECTORS  # 1/Angstrom
    temperature: float = 0.3                            # K
    tc: float = TC_UTE2                                 # K — uniform superconductivity
    t_pdw: float = T_PDW_UTE2                           # K — PDW onset
    field: tuple[float, float, float] = (0.0, 0.0, 0.0)  # Tesla, (x, y, z)
    hc2: tuple[float, float, float] = HC2_UTE2          # Tesla at T = 0
    h_pdw: tuple[float, float, float] = H_PDW_UTE2      # Tesla at T = 0
    h_rot: tuple[float, float, float] = H_ROT_UTE2      # Tesla — d-vector rotation onset
    delta0_mev: float = DELTA0_UTE2                     # meV — uniform gap at T = 0
    pdw_ratio: float = PDW_RATIO_UTE2                   # Delta_p(0) / Delta_0(0)
    overlap0: tuple[float, float, float] = OVERLAP0_UTE2  # d0 . d_{p_i} at B = 0, in [0, 1]
    phases: tuple[float, float, float] = (0.0, 0.0, 0.0)  # PDW sliding phases (rad)
    kappa_p: float = PDW_KAPPA_P                        # p-CDW coupling g1/r1 (scaled)
    kappa_q: float = PDW_KAPPA_Q                        # q-CDW coupling g2/r2 (scaled)
    kappa_h: float = PDW_KAPPA_H                        # harmonic coupling gamma2/r4
    kappa_harm: float = PDW_KAPPA_HARM                  # p-harmonic feed gamma1/r2
    floor: float = PDW_VISIBILITY_FLOOR                 # |rho|^2 visibility threshold


def moire_locked_config(
    moire_period: float,
    twist_angle_deg: float = 0.0,
    delta0_mev: float = DELTA_AVG,
    pdw_ratio: float | None = None,
) -> PdwConfig:
    """PDW model locked to a heterostructure's moire period (what-if preset).

    The CPDM of the FeTe/TI heterostructures is imprinted by the moire, not
    spontaneous, so the preset sets ``T_PDW = Tc``: every composite order
    vanishes at Tc.  Raising ``t_pdw`` above ``tc`` asks what an intrinsic,
    UTe2-like PDW at the moire wavevector would add — a q = 2Q_m charge
    modulation (half the moire period) surviving above Tc.  Singlet-like
    pairing: all d-vectors aligned, isotropic Bc2 for both orders.

    ``pdw_ratio`` defaults to :func:`moire_locked_pdw_ratio` for the
    configured gaps and coherence length.
    """
    if pdw_ratio is None:
        pdw_ratio = moire_locked_pdw_ratio(moire_period)
    isotropic = (BC2_FETE, BC2_FETE, BC2_FETE)
    return PdwConfig(
        p_vectors=tuple(
            (float(kx), float(ky))
            for kx, ky in moire_locked_wavevectors(moire_period, twist_angle_deg)
        ),
        temperature=0.3 * TC_FETE_FILM,
        tc=TC_FETE_FILM,
        t_pdw=TC_FETE_FILM,
        hc2=isotropic,
        h_pdw=isotropic,
        h_rot=isotropic,
        delta0_mev=delta0_mev,
        pdw_ratio=pdw_ratio,
        overlap0=(1.0, 1.0, 1.0),
    )


def _validate(config: PdwConfig) -> None:
    p = np.asarray(config.p_vectors, dtype=float)
    if p.shape != (3, 2) or not np.all(np.isfinite(p)):
        raise ValueError("p_vectors must be three finite 2D wavevectors")
    if not np.isfinite(config.temperature) or config.temperature < 0:
        raise ValueError("temperature must be finite and >= 0")
    if not config.tc > 0 or not config.t_pdw > 0:
        raise ValueError("tc and t_pdw must be positive")
    for name in ("field", "phases"):
        vals = np.asarray(getattr(config, name), dtype=float)
        if vals.shape != (3,) or not np.all(np.isfinite(vals)):
            raise ValueError(f"{name} must be three finite numbers")
    for name in ("hc2", "h_pdw", "h_rot"):
        vals = np.asarray(getattr(config, name), dtype=float)
        if vals.shape != (3,) or not np.all(vals > 0):
            raise ValueError(f"{name} must be three positive fields")
    if not config.delta0_mev >= 0 or not config.pdw_ratio >= 0:
        raise ValueError("delta0_mev and pdw_ratio must be >= 0")
    c0 = np.asarray(config.overlap0, dtype=float)
    if c0.shape != (3,) or not np.all((c0 >= 0) & (c0 <= 1)):
        raise ValueError("overlap0 entries must lie in [0, 1]")
    couplings = (config.kappa_p, config.kappa_q, config.kappa_h, config.kappa_harm)
    if not np.all(np.isfinite(couplings)):
        raise ValueError("kappa couplings must be finite")
    if not config.floor >= 0:
        raise ValueError("floor must be >= 0")


def _check_critical(critical) -> np.ndarray:
    h = np.asarray(critical, dtype=float)
    # `not all(h > 0)` also rejects NaN, matching the Rust validation.
    if not np.all(h > 0):
        raise ValueError("critical fields must be positive")
    return h


def _unit(direction) -> np.ndarray:
    u = np.asarray(direction, dtype=float)
    norm = float(np.linalg.norm(u))
    if u.shape != (3,) or not np.isfinite(norm) or norm < NORM_FLOOR:
        raise ValueError("direction must be a nonzero 3-vector")
    return u / norm


# ---------------------------------------------------------------------------
# Mean-field amplitudes and d-vector geometry
# ---------------------------------------------------------------------------


def reduced_field(field, critical) -> float:
    """Ellipsoidal reduced field h = sqrt(sum_n (B_n / H_n)^2)."""
    b = np.asarray(field, dtype=float)
    h = _check_critical(critical)
    return float(np.sqrt(np.sum((b / h) ** 2)))


def landau_amplitude(temperature: float, t_star: float, reduced: float) -> float:
    """Normalised mean-field amplitude sqrt(max(0, 1 - T/T* - h)).

    The Landau minimum |d|^2 = a (T* - T) / (2 l) with a field-shifted
    transition T*(B) = T* (1 - h).
    """
    if not t_star > 0:
        raise ValueError("t_star must be positive")
    return float(np.sqrt(max(0.0, 1.0 - temperature / t_star - reduced)))


def critical_field_along(t_star: float, critical, temperature: float, direction) -> float:
    """Field magnitude along ``direction`` that kills the order at ``temperature`` (T)."""
    if not t_star > 0:
        raise ValueError("t_star must be positive")
    u = _unit(direction)
    h = _check_critical(critical)
    return max(0.0, 1.0 - temperature / t_star) / float(np.sqrt(np.sum((u / h) ** 2)))


def onset_field_along(h_rot, direction) -> float:
    """d-vector rotation onset field along ``direction`` (Tesla)."""
    return critical_field_along(1.0, h_rot, 0.0, direction)


def dvector_overlaps(field, h_rot, overlap0) -> np.ndarray:
    """d0 . d_{p_i} after field-induced rotation, shape ``(3,)``.

    Below the onset (h_rot <= 1) the zero-field overlaps hold; above it the
    orthogonal part shrinks as 1/h_rot, so an initially orthogonal pair
    reaches overlap sqrt(1 - 1/h_rot^2).
    """
    c0 = np.asarray(overlap0, dtype=float)
    if not np.all((c0 >= 0) & (c0 <= 1)):
        raise ValueError("overlap0 entries must lie in [0, 1]")
    h = reduced_field(field, h_rot)
    shrink = 1.0 if h <= 1.0 else 1.0 / h
    s = np.sqrt(1.0 - c0**2) * shrink
    return np.sqrt(1.0 - s**2)


def classify_regime(delta0: float, delta_p: float) -> str:
    """Name the ordered phase from the two normalised amplitudes."""
    if delta_p > 0 and delta0 > 0:
        return "coexistence"
    if delta_p > 0:
        return "pdw_only"
    if delta0 > 0:
        return "uniform_only"
    return "normal"


# ---------------------------------------------------------------------------
# Composite orders
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PdwState:
    """Mean-field PDW state and its composite orders at one (T, B)."""

    delta0: float         # |d_0(T, B)| / |d_0(0)|
    delta_p: float        # |d_p(T, B)| / |d_p(0)|
    overlap: np.ndarray   # (3,) d0 . d_{p_i} after field rotation
    regime: str           # one of PDW_REGIMES
    rho_p: np.ndarray     # (3,) complex — p_i CDW (PDW x uniform SC)
    rho_q: np.ndarray     # (3,) complex — q_i = 2 p_i CDW (PDW alone)
    rho_h: np.ndarray     # (2,) complex — h_1, h_2 harmonics
    spin_p: np.ndarray    # (3,) complex — SPECULATIVE composite SDW at p_i


def compute_pdw_state(config: PdwConfig) -> PdwState:
    """Evaluate the mean-field amplitudes and every composite order."""
    _validate(config)
    t = config.temperature
    d0 = landau_amplitude(t, config.tc, reduced_field(config.field, config.hc2))
    dp = landau_amplitude(t, config.t_pdw, reduced_field(config.field, config.h_pdw))
    c = dvector_overlaps(config.field, config.h_rot, config.overlap0)
    s = np.sqrt(np.maximum(1.0 - c**2, 0.0))
    phase = np.exp(1j * np.asarray(config.phases, dtype=float))

    # Stationary points of F in the CDW amplitudes: rho = -(source)* / r.
    rho_p = -2.0 * config.kappa_p * d0 * dp * c * phase
    rho_q = -config.kappa_q * dp**2 * phase**2 - config.kappa_harm * rho_p**2
    rho_h = -config.kappa_h * rho_q[2] * np.conj(rho_q[[1, 0]])
    spin_p = 2.0 * config.kappa_p * d0 * dp * s * phase

    return PdwState(
        delta0=d0,
        delta_p=dp,
        overlap=c,
        regime=classify_regime(d0, dp),
        rho_p=rho_p,
        rho_q=rho_q,
        rho_h=rho_h,
        spin_p=spin_p,
    )


@dataclass(frozen=True)
class PdwPeak:
    """One composite-order Fourier peak (the +k member of a +-k pair)."""

    label: str        # "p1" ... "h2"
    family: str       # "p", "q" or "h"
    kx: float         # 1/Angstrom
    ky: float         # 1/Angstrom
    amplitude: float  # |rho_k|
    visible: bool     # |rho_k|^2 above the visibility floor

    @property
    def intensity(self) -> float:
        return self.amplitude**2


def _composite_terms(
    config: PdwConfig, state: PdwState
) -> list[tuple[str, str, np.ndarray, complex]]:
    """(label, family, k, rho_k) for every composite CDW component."""
    p = np.asarray(config.p_vectors, dtype=float)
    terms = [(f"p{i + 1}", "p", p[i], state.rho_p[i]) for i in range(3)]
    terms += [(f"q{i + 1}", "q", 2.0 * p[i], state.rho_q[i]) for i in range(3)]
    terms += [
        (f"h{j + 1}", "h", h, state.rho_h[j]) for j, h in enumerate(harmonic_vectors(p))
    ]
    return terms


def pdw_peaks(config: PdwConfig, state: PdwState) -> list[PdwPeak]:
    """The eight composite CDW peaks (p1-3, q1-3, h1-2) with amplitudes."""
    return [
        PdwPeak(
            label=label,
            family=family,
            kx=float(k[0]),
            ky=float(k[1]),
            amplitude=float(abs(rho)),
            visible=bool(abs(rho) ** 2 > config.floor),
        )
        for label, family, k, rho in _composite_terms(config, state)
    ]


# ---------------------------------------------------------------------------
# Real-space fields
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PdwFields:
    """Real-space maps of the PDW state on a grid."""

    pair_amplitude: np.ndarray  # 2D |d(r)| in meV
    charge_density: np.ndarray  # 2D rho(r), dimensionless
    spin_density: np.ndarray    # 2D S_z(r), dimensionless — SPECULATIVE


def pdw_fields_at(config: PdwConfig, state: PdwState, x, y):
    """Evaluate (|d|, rho, S_z) at points (x, y); arrays broadcast.

    The pair field is d(r) = Delta_0 n_0 + sum_i 2 Delta_p n_i cos(p_i.r + phi_i)
    with n_0 = (1, 0) and n_i = (c_i, s_i) in the plane the d-vectors rotate
    in, so |d| is the local gap of a unitary triplet state.  Densities are
    rho(r) = sum_k 2 Re[rho_k exp(i k.r)] and likewise for S_z at p_i.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    p = np.asarray(config.p_vectors, dtype=float)
    phases = np.asarray(config.phases, dtype=float)
    shape = np.broadcast(x, y).shape
    delta0 = config.delta0_mev * state.delta0
    delta_p = config.delta0_mev * config.pdw_ratio * state.delta_p
    s = np.sqrt(np.maximum(1.0 - state.overlap**2, 0.0))

    d_par = np.full(shape, delta0)
    d_perp = np.zeros(shape)
    spin = np.zeros(shape)
    for i in range(3):
        arg = p[i, 0] * x + p[i, 1] * y
        wave = np.cos(arg + phases[i])
        d_par += 2.0 * delta_p * state.overlap[i] * wave
        d_perp += 2.0 * delta_p * s[i] * wave
        spin += 2.0 * np.real(state.spin_p[i] * np.exp(1j * arg))

    charge = np.zeros(shape)
    for _label, _family, k, rho in _composite_terms(config, state):
        charge += 2.0 * np.real(rho * np.exp(1j * (k[0] * x + k[1] * y)))

    return np.hypot(d_par, d_perp), charge, spin


def pdw_real_space(config: PdwConfig, state: PdwState, x: np.ndarray, y: np.ndarray) -> PdwFields:
    """Pair amplitude, charge density and spin density on the (x, y) grid."""
    X, Y = np.meshgrid(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    pair, charge, spin = pdw_fields_at(config, state, X, Y)
    return PdwFields(pair_amplitude=pair, charge_density=charge, spin_density=spin)


# ---------------------------------------------------------------------------
# Sweeps and phase diagram
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PdwSweep:
    """Order parameters and peak intensities |rho_k|^2 along a 1D sweep."""

    values: np.ndarray       # swept parameter (K or Tesla)
    delta0: np.ndarray       # (n,)
    delta_p: np.ndarray      # (n,)
    intensity_p: np.ndarray  # (3, n)
    intensity_q: np.ndarray  # (3, n)
    intensity_h: np.ndarray  # (2, n)


def _sweep(configs: list[PdwConfig], values: np.ndarray) -> PdwSweep:
    states = [compute_pdw_state(c) for c in configs]

    def intensities(name: str, width: int) -> np.ndarray:
        # reshape keeps the (width, n) layout for an empty sweep too.
        rows = np.array([getattr(s, name) for s in states], dtype=complex)
        return np.abs(rows.reshape(len(states), width)).T ** 2

    return PdwSweep(
        values=values,
        delta0=np.array([s.delta0 for s in states], dtype=float),
        delta_p=np.array([s.delta_p for s in states], dtype=float),
        intensity_p=intensities("rho_p", 3),
        intensity_q=intensities("rho_q", 3),
        intensity_h=intensities("rho_h", 2),
    )


def temperature_sweep(config: PdwConfig, temperatures) -> PdwSweep:
    """Sweep temperature at the configured field."""
    temps = np.asarray(temperatures, dtype=float)
    return _sweep([replace(config, temperature=float(t)) for t in temps], temps)


def field_sweep(config: PdwConfig, magnitudes, direction) -> PdwSweep:
    """Sweep |B| along ``direction`` at the configured temperature."""
    u = _unit(direction)
    mags = np.asarray(magnitudes, dtype=float)
    return _sweep(
        [replace(config, field=tuple(float(v) for v in b * u)) for b in mags], mags
    )


def regime_map(config: PdwConfig, temperatures, magnitudes, direction) -> np.ndarray:
    """Regime index into ``PDW_REGIMES`` on a (T, |B|) grid, shape ``(n_T, n_B)``."""
    _validate(config)
    u = _unit(direction)
    t = np.asarray(temperatures, dtype=float)[:, None]
    b = np.abs(np.asarray(magnitudes, dtype=float))[None, :]
    h_sc = b * np.sqrt(np.sum((u / np.asarray(config.hc2)) ** 2))
    h_pdw = b * np.sqrt(np.sum((u / np.asarray(config.h_pdw)) ** 2))
    sc = 1.0 - t / config.tc - h_sc > 0
    pdw = 1.0 - t / config.t_pdw - h_pdw > 0
    return np.where(pdw, np.where(sc, 3, 1), np.where(sc, 2, 0))


# ---------------------------------------------------------------------------
# STM-style spectrum (presentation helper)
# ---------------------------------------------------------------------------


def charge_spectrum(
    config: PdwConfig,
    state: PdwState,
    half_width: float,
    n: int = 256,
    pad_to: int = 1024,
    k_lim: float | None = None,
) -> dict:
    """STM-style spectrum of rho(r): Hann taper, zero padding, cropped to |k| <= k_lim.

    rho is re-evaluated on its own ``n x n`` grid over
    ``[-half_width, half_width]``, so the k-resolution 2*pi / (pad_to * dx)
    does not depend on the display grid.  Without the taper, incommensurate
    peaks leak into streaks along kx and ky.  ``k_lim`` defaults to 1.3x the
    largest composite wavevector.  Returns ``kx``, ``ky`` (1/Angstrom) and the
    log10 ``power_spectrum`` like :func:`~waytogocoop.computation.fourier.fft_2d`.
    The desktop app transforms its display grid instead (no padding or crop).
    """
    if not half_width > 0 or n < 8:
        raise ValueError("half_width must be positive and n >= 8")
    x = np.linspace(-half_width, half_width, n)
    X, Y = np.meshgrid(x, x)
    _pair, charge, _spin = pdw_fields_at(config, state, X, Y)
    taper = np.outer(np.hanning(n), np.hanning(n))
    spectrum = fft_2d(np.pad(charge * taper, (0, max(pad_to - n, 0))), x[1] - x[0])
    if k_lim is None:
        k_lim = 1.3 * max(
            float(np.hypot(k[0], k[1])) for _l, _f, k, _r in _composite_terms(config, state)
        )
    keep_x = np.abs(spectrum["kx"]) <= k_lim
    keep_y = np.abs(spectrum["ky"]) <= k_lim
    return {
        "kx": spectrum["kx"][keep_x],
        "ky": spectrum["ky"][keep_y],
        "power_spectrum": spectrum["power_spectrum"][np.ix_(keep_y, keep_x)],
    }

