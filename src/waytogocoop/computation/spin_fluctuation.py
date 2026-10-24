"""Spin-fluctuation pairing: LCCO magnetotransport and what-if pairing routes.

SPECULATIVE as a module: LCCO is not one of this project's heterostructures,
and everything here is normal-state electron transport, not Cooper-pair
transport.  Duffy et al. use the transport to *measure* the pairing glue.

Part A — LCCO transport, paper-faithful (Duffy et al., Nat. Phys. (2026),
doi:10.1038/s41567-026-03454-6; arXiv:2502.13612, page numbers from the arXiv
v1 PDF).  Boltzmann (Shockley-Chambers) magnetotransport of electron-doped
La2-xCexCuO4 with the scattering-rate ansatz of Eq. 1, the analytic Fermi
surface of Eqs. 7-10 and the Jones-Zener Hall coefficient with Kontani
current-vertex corrections (CVCs, Eqs. 11-15).  It reproduces the paper's
Fig. 4-5 simulations; one inferred reading, the hot-spot height of Fig. 4b1
(31.4 ps^-1 where Table 2 prints 50).  The anisotropic T-linear scattering
strength g(x) alpha1 defines lambda_tr, the Fermi-surface-averaged T-linear
coupling (Allen's form applied to the microscopic rate), which tracks Tc.

Part B — SPECULATIVE pairing map Tc(lambda): an EMPIRICAL power law through
the paper's Fig. 5c Tc labels; the McMillan form is included as a failing
comparison (a phonon formula; its best fit pins mu* at 0 and misses the dome
edge).

Part C — SPECULATIVE moire glue-modulation what-if.  lambda(r) = lambda_bar
(1 + eta m~(r)) on the project's moire pattern, m~ = -cos(pi m), gives the
local gap Delta_loc(r) = Delta_bar Tc(lambda(r)) / Tc(lambda_bar) with
Delta_bar = DELTA_AVG Tc* / TC_FETE_FILM.  It is dressed by the CPDM
coherence factor F = exp(-xi / L_m), Delta = Delta_bar + F (Delta_loc -
Delta_bar), with xi(Tc*) from clean or dirty scaling and L_m the shared
two-lattice moire period; "local" (F = 1) is the L_m >> xi limit.  The LCCO
map supplies only the *shape* of Tc(lambda); it is not a model of FeTe, whose
AFM is bicollinear and whose proposed pairing is s+- via (pi, 0)-type
fluctuations.

Part D — contrast curve: hole-doped H/T quadrature scaling (Ayres et al.,
Nature 595, 661 (2021)), a phenomenological form, not a model.

Conventions: phi is the polar angle of k - M about the hole pocket centre
M = (pi/a, pi/a), counter-clockwise from +kx (phi = 0 antinode, 45 deg node).
Carriers are holes (+e) with the outward velocity of Eq. 9; for B > 0 the
past orbit runs to increasing phi and R_H > 0.  d = c/2.  Rates are in
ps^-1 at the API (s^-1 inside), resistivities in micro-Ohm cm, R_H in m^3/C.

Mirrors crates/moire-core/src/spin_fluctuation.rs — changes here must land in both.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

from waytogocoop.computation.pdw import moire_locked_period
from waytogocoop.config import (
    A_LCCO,
    ALPHA1_LCCO,
    ALPHA2_LCCO,
    ANGSTROM_TO_M,
    DEFAULT_COHERENCE_LENGTH,
    DELTA_AMPLITUDE,
    DELTA_AVG,
    ELEMENTARY_CHARGE_C,
    G_TABLE_LCCO,
    GAMMA_IMP_LCCO,
    HBAR_J_S,
    HOTSPOT_PEAK_LCCO,
    HOTSPOT_WIDTH_LCCO,
    KB_J_K,
    KF0_LCCO,
    KF_ANISOTROPY_LCCO,
    LAYER_SPACING_LCCO,
    PHI_HS_DEG_LCCO,
    SF_CVC_EPSILON,
    SF_CVC_SIGMA,
    SF_DOPING_REF,
    SF_FIELD_MAX,
    SF_FIELD_STEP,
    SF_FIT_FIELD,
    SF_FIT_TEMPERATURE,
    SF_GLUE_COHERENCE,
    SF_GLUE_ETA,
    SF_MCMILLAN_MU_STAR,
    SF_MCMILLAN_OMEGA_K,
    SF_MR_TEMPERATURES,
    SF_N_PHI,
    SF_NU,
    SF_POWER_LAW_EXPONENT,
    SF_POWER_LAW_LAMBDA_C,
    SF_POWER_LAW_LAMBDA_REF,
    SF_POWER_LAW_TC_REF,
    SF_QUAD_A,
    SF_QUAD_ALPHA_KB,
    SF_QUAD_B,
    SF_QUAD_GAMMA_MUB,
    SF_QUAD_RHO0,
    SF_RHO_FIT_RANGE,
    SF_SLOPE_FIELD,
    TC_FETE_FILM,
    TC_LABELS_FIG5C,
    TC_TABLE_LCCO,
    VF0_LCCO,
    VF_ANISOTROPY_LCCO,
    X_FSR_LCCO,
)
from waytogocoop.materials.database import get_material

SF_INELASTIC_FORMS: tuple[str, ...] = ("afm_boundary", "isotropic", "antinodal", "none")
SF_DOPING_REGIMES: tuple[str, ...] = ("fs_reconstruction", "superconducting", "beyond_dome")
SF_PAIRING_FORMS: tuple[str, ...] = ("power_law", "mcmillan")
SF_COHERENCE_MODELS: tuple[str, ...] = ("clean", "dirty", "local")
DOPINGS_LCCO: tuple[float, ...] = tuple(x for x, _ in G_TABLE_LCCO)  # 19, ascending

# Default L_m of the glue what-if: the shared two-lattice period of the default
# FeTe / Sb2Te3 stack at zero twist (36.68576576576573 Angstrom).  Config cannot
# import pdw, so it is computed here; the Rust twin pins the literal and tests it.
SF_GLUE_MOIRE_PERIOD: float = moire_locked_period(
    get_material("FeTe").a, get_material("Sb2Te3").a, 0.0
)

_PS_INV = 1.0e12  # ps^-1 -> s^-1
_OHM_M_TO_MUOHM_CM = 1.0e8
_DOPING_TOL = 1.0e-12  # table nodes are matched to this
_CROSSING_STEPS = 100  # bisection steps for the zone-boundary crossing
_SERIES_TERMS = 28  # E_n series terms below R = 2
_SERIES_SWITCH = 2.0
_N_MOMENTS = 7  # E_0 .. E_6


def _real(name: str, value) -> float:
    """``value`` as a finite float; NaN, infinities and non-numbers raise ValueError.

    Only real scalars pass: float() would also parse "27" or Decimal("27"), and the
    config validators check a field but compute with its raw value.
    """
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite number")
    try:
        v = float(value)
    except OverflowError:  # an int beyond the float range
        raise ValueError(f"{name} must be a finite number") from None
    if not math.isfinite(v):
        raise ValueError(f"{name} must be a finite number")
    return v


def _positive(name: str, value) -> float:
    v = _real(name, value)
    if not v > 0.0:
        raise ValueError(f"{name} must be > 0")
    return v


def _non_negative(name: str, value) -> float:
    v = _real(name, value)
    if not v >= 0.0:
        raise ValueError(f"{name} must be >= 0")
    return v


def _is_int(value) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _temperature_array(temperatures) -> np.ndarray:
    try:
        t = np.atleast_1d(np.asarray(temperatures, dtype=float))
    except (TypeError, ValueError):
        raise ValueError("temperatures must be finite numbers >= 0") from None
    if t.ndim != 1 or not np.all(np.isfinite(t)) or np.any(t < 0.0):
        raise ValueError("temperatures must be finite numbers >= 0")
    return t


def _temperature(value) -> np.ndarray:
    """One temperature (finite, >= 0) as a one-element array."""
    return np.array([_non_negative("temperature", value)])


def _field_array(fields) -> np.ndarray:
    try:
        b = np.atleast_1d(np.asarray(fields, dtype=float))
    except (TypeError, ValueError):
        raise ValueError("fields must be finite numbers") from None
    if b.ndim != 1 or not np.all(np.isfinite(b)):
        raise ValueError("fields must be finite numbers")
    return b


def _finite_array(name: str, values) -> np.ndarray:
    try:
        a = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be finite") from None
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite")
    return a


def _scalar_or_array(values: np.ndarray):
    return float(values) if values.ndim == 0 else values


# ---------------------------------------------------------------------------
# Part A — LCCO transport (paper-faithful)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SfConfig:
    """LCCO transport model (Duffy et al. 2026): Table 2 at x = 0.159, effective hot spot."""

    doping: float = SF_DOPING_REF  # x in [0.116, 0.188] (g interpolated)
    inelastic: str = "afm_boundary"  # SF_INELASTIC_FORMS
    hotspots: bool = True
    gamma_imp: float = GAMMA_IMP_LCCO  # ps^-1, > 0
    alpha1: float = ALPHA1_LCCO  # ps^-1 K^-1, >= 0 (times g(x))
    alpha2: float = ALPHA2_LCCO  # ps^-1 K^-2, >= 0
    hotspot_peak: float = HOTSPOT_PEAK_LCCO  # ps^-1, >= 0
    hotspot_width: float = HOTSPOT_WIDTH_LCCO  # rad, 1/e half-width, > 0
    nu: int = SF_NU  # even, 2..24
    phi_hs_deg: float = PHI_HS_DEG_LCCO  # (0, 45)
    kf0: float = KF0_LCCO  # 1/m
    kf_anisotropy: float = KF_ANISOTROPY_LCCO  # [0, 1)
    vf0: float = VF0_LCCO  # m/s
    vf_anisotropy: float = VF_ANISOTROPY_LCCO  # [0, 1)
    layer_spacing: float = LAYER_SPACING_LCCO  # Angstrom, > 0
    cvc_epsilon: float = SF_CVC_EPSILON  # [0, 1]
    cvc_sigma: float = SF_CVC_SIGMA  # rad, > 0
    n_phi: int = SF_N_PHI  # power of two in [64, 8192]


_DEFAULT_CONFIG = SfConfig()  # frozen, so one shared default instance is safe


@dataclass(frozen=True)
class SfFermiSurface:
    """The analytic Fermi surface (Eqs. 7-10) on the n_phi grid."""

    phi: np.ndarray  # (N,) rad, 2 pi m / N
    kf: np.ndarray  # (N,) 1/m, Eq. 7
    vf: np.ndarray  # (N,) m/s, Eq. 8
    gamma: np.ndarray  # (N,) rad, tan(gamma) = d ln kF / d phi (Eq. 10)
    vx: np.ndarray  # (N,) m/s, vF cos(phi - gamma) (Eq. 9, outward)
    vy: np.ndarray  # (N,) m/s, vF sin(phi - gamma)
    weight: np.ndarray  # (N,) s/m^2, kF / (vF cos gamma): DOS weight |dk/dphi| / vF
    omega_per_tesla: np.ndarray  # (N,) rad s^-1 T^-1, e vF cos(gamma) / (hbar kF)


@dataclass(frozen=True)
class SfMagnetoresistance:
    """rho(H, T) family on the field grid; arrays are (n_temperatures, n_fields)."""

    temperatures: np.ndarray  # (nT,) K
    fields: np.ndarray  # (nH,) T, 0 .. field_max inclusive
    rho_xx: np.ndarray  # micro-Ohm cm
    rho_yx: np.ndarray  # micro-Ohm cm
    mr: np.ndarray  # rho_xx(H) / rho_xx(0) - 1
    drho_dh: np.ndarray  # micro-Ohm cm / T (field_derivative stencil)
    h_over_t: np.ndarray  # T / K


@dataclass(frozen=True)
class SfResistivityFit:
    """Least-squares rho0 + A1 T + A2 T^2 on [t_min, t_max]."""

    rho0: float  # micro-Ohm cm
    a1: float  # micro-Ohm cm / K
    a2: float  # micro-Ohm cm / K^2
    t_min: float  # K
    t_max: float  # K


@dataclass(frozen=True)
class SfDopingSweep:
    """Per tabulated doping (19 entries, ascending): coupling and transport at (T, H)."""

    dopings: np.ndarray  # x
    g: np.ndarray  # g(x), Tables 3-4
    tc: np.ndarray  # K, TC_TABLE_LCCO
    lambda_tr: np.ndarray  # FS-averaged T-linear coupling
    lambda_rho: np.ndarray  # Allen-Drude coupling from the model's own rho(T)
    rho0: np.ndarray  # rho(0, temperature), micro-Ohm cm
    mr: np.ndarray  # MR(field, temperature)
    temperature: float  # K
    field: float  # T


def _validate(config: SfConfig) -> None:
    doping = _real("doping", config.doping)
    if not DOPINGS_LCCO[0] - _DOPING_TOL <= doping <= DOPINGS_LCCO[-1] + _DOPING_TOL:
        raise ValueError("doping must lie in the tabulated range [0.116, 0.188]")
    if config.inelastic not in SF_INELASTIC_FORMS:
        raise ValueError(f"inelastic must be one of {SF_INELASTIC_FORMS}")
    if not isinstance(config.hotspots, (bool, np.bool_)):
        raise ValueError("hotspots must be a bool")
    _positive("gamma_imp", config.gamma_imp)
    for name in ("alpha1", "alpha2", "hotspot_peak"):
        _non_negative(name, getattr(config, name))
    for name in ("hotspot_width", "cvc_sigma", "kf0", "vf0", "layer_spacing"):
        _positive(name, getattr(config, name))
    if not _is_int(config.nu) or config.nu % 2 or not 2 <= config.nu <= 24:
        raise ValueError("nu must be an even integer in [2, 24]")
    if not 0.0 < _real("phi_hs_deg", config.phi_hs_deg) < 45.0:
        raise ValueError("phi_hs_deg must lie in (0, 45)")
    for name in ("kf_anisotropy", "vf_anisotropy"):
        if not 0.0 <= _real(name, getattr(config, name)) < 1.0:
            raise ValueError(f"{name} must lie in [0, 1)")
    if not 0.0 <= _real("cvc_epsilon", config.cvc_epsilon) <= 1.0:
        raise ValueError("cvc_epsilon must lie in [0, 1]")
    n = config.n_phi
    if not _is_int(n) or not 64 <= n <= 8192 or n & (n - 1):
        raise ValueError("n_phi must be a power of two in [64, 8192]")


def g_of_x(doping: float) -> float:
    """Anisotropic T-linear strength g(x), Tables 3-4; linear between table points.

    Exact at the 19 tabulated dopings (|dx| <= 1e-12); ValueError outside
    [0.116, 0.188].
    """
    x = _real("doping", doping)
    if not DOPINGS_LCCO[0] - _DOPING_TOL <= x <= DOPINGS_LCCO[-1] + _DOPING_TOL:
        raise ValueError("doping must lie in the tabulated range [0.116, 0.188]")
    segments = zip(G_TABLE_LCCO[:-1], G_TABLE_LCCO[1:], strict=True)
    (x0, g0), (x1, g1) = next(seg for seg in segments if x <= seg[1][0] + _DOPING_TOL)
    if abs(x - x0) <= _DOPING_TOL:
        return g0
    if abs(x - x1) <= _DOPING_TOL:
        return g1
    return g0 + (g1 - g0) * (x - x0) / (x1 - x0)


def tc_of_x(doping: float) -> float:
    """Tc(x) in K: piecewise linear through TC_TABLE_LCCO, clamped >= 0 and flat outside."""
    x = _real("doping", doping)
    table = TC_TABLE_LCCO
    if x <= table[0][0]:
        return table[0][1]
    for (x0, t0), (x1, t1) in zip(table[:-1], table[1:], strict=True):
        if x <= x1:
            return max(0.0, t0 + (t1 - t0) * (x - x0) / (x1 - x0))
    return table[-1][1]


def snap_doping(doping) -> float:
    """Nearest tabulated x (ties go to the lower x); non-finite or garbage -> SF_DOPING_REF."""
    try:
        x = float(doping)
    except (TypeError, ValueError):
        return SF_DOPING_REF
    if not math.isfinite(x):
        return SF_DOPING_REF
    return min(DOPINGS_LCCO, key=lambda xt: abs(xt - x))


def doping_regime(doping: float) -> str:
    """One of SF_DOPING_REGIMES: FS reconstruction below x_FSR, then Tc(x) > 0 or not."""
    x = _real("doping", doping)
    if x < X_FSR_LCCO:
        return "fs_reconstruction"
    return "superconducting" if tc_of_x(x) > 0.0 else "beyond_dome"


def film_of_x(doping: float) -> str:
    """The paper's film: "S1" (Table 3) for x <= 0.159, "S2" (Table 4) above."""
    return "S1" if _real("doping", doping) <= SF_DOPING_REF else "S2"


def _fs_at(phi, config: SfConfig) -> tuple[np.ndarray, ...]:
    phi = np.asarray(phi, dtype=float)
    s2 = np.sin(2.0 * phi) ** 2
    kf = config.kf0 * (1.0 + config.kf_anisotropy * s2)
    vf = config.vf0 * (1.0 + config.vf_anisotropy * s2)
    gamma = np.arctan(
        2.0 * config.kf_anisotropy * np.sin(4.0 * phi) / (1.0 + config.kf_anisotropy * s2)
    )
    vx = vf * np.cos(phi - gamma)
    vy = vf * np.sin(phi - gamma)
    return kf, vf, gamma, vx, vy


def fermi_surface_at(phi, config: SfConfig = _DEFAULT_CONFIG) -> tuple[np.ndarray, ...]:
    """(kF, vF, gamma, vx, vy) at arbitrary angles phi (Eqs. 7-10; 1/m, m/s, rad)."""
    _validate(config)
    return _fs_at(_finite_array("phi", phi), config)


def _fermi_surface(config: SfConfig) -> SfFermiSurface:
    n = config.n_phi
    phi = 2.0 * np.pi * np.arange(n) / n
    kf, vf, gamma, vx, vy = _fs_at(phi, config)
    return SfFermiSurface(
        phi=phi,
        kf=kf,
        vf=vf,
        gamma=gamma,
        vx=vx,
        vy=vy,
        weight=kf / (vf * np.cos(gamma)),
        omega_per_tesla=ELEMENTARY_CHARGE_C * vf * np.cos(gamma) / (HBAR_J_S * kf),
    )


def fermi_surface(config: SfConfig = _DEFAULT_CONFIG) -> SfFermiSurface:
    """The Fermi surface on the grid phi_m = 2 pi m / n_phi."""
    _validate(config)
    return _fermi_surface(config)


def luttinger_holes(config: SfConfig = _DEFAULT_CONFIG) -> float:
    """Holes per Cu from the pocket area, a^2 <kF^2> / (2 pi) (spin included)."""
    _validate(config)
    kf = _fermi_surface(config).kf
    a = A_LCCO * ANGSTROM_TO_M
    return float(a * a * np.mean(kf * kf) / (2.0 * np.pi))


def _zone_boundary_crossing(config: SfConfig) -> float:
    """Root of kF(phi) (cos phi + sin phi) = pi/a in (0, pi/4), radians (bisection midpoint)."""
    q = np.pi / (A_LCCO * ANGSTROM_TO_M)

    def mismatch(p: float) -> float:
        kf = _fs_at(np.array([p]), config)[0][0]
        return kf * (math.cos(p) + math.sin(p)) - q

    lo, hi = 0.0, np.pi / 4.0
    if not mismatch(lo) < 0.0 < mismatch(hi):
        raise ValueError("the Fermi surface does not cross the AFM zone boundary in (0, 45) deg")
    for _ in range(_CROSSING_STEPS):
        mid = 0.5 * (lo + hi)
        if mismatch(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def zone_boundary_crossing_deg(config: SfConfig = _DEFAULT_CONFIG) -> float:
    """Where the FS meets the AFM zone boundary |kx| + |ky| = pi/a (27.20 deg by default).

    100-step bisection on (0, 45) deg; ValueError if the bracket has no sign change.
    """
    _validate(config)
    return math.degrees(_zone_boundary_crossing(config))


def _eight_angles(p: float) -> np.ndarray:
    """[p, pi/2-p, p+pi/2, pi-p, p+pi, 3pi/2-p, p+3pi/2, 2pi-p]."""
    return np.array([b + m * 0.5 * np.pi for m in range(4) for b in (p, 0.5 * np.pi - p)])


def hotspot_angles(config: SfConfig = _DEFAULT_CONFIG) -> np.ndarray:
    """The eight hot-spot centres phi_n of Gamma_HS (rad), built on phi_HS."""
    _validate(config)
    return _eight_angles(math.radians(config.phi_hs_deg))


def cvc_centres(config: SfConfig = _DEFAULT_CONFIG) -> np.ndarray:
    """The eight CVC centres (rad): the FS x zone-boundary crossing, same order as the hot spots.

    CVCs need k and k + Q both on the FS, which holds only at the crossing, so
    eps(phi) is centred there; phi_HS moves only Gamma_HS and W.  ValueError if
    the FS misses the zone boundary.
    """
    _validate(config)
    return _eight_angles(_zone_boundary_crossing(config))


def _wrap(d):
    return np.mod(d + np.pi, 2.0 * np.pi) - np.pi


def _inelastic_weight(phi: np.ndarray, config: SfConfig) -> np.ndarray:
    if config.inelastic == "afm_boundary":
        off = 0.25 * np.pi - math.radians(config.phi_hs_deg)
        return np.sin(2.0 * (phi + off)) ** config.nu + np.sin(2.0 * (phi - off)) ** config.nu
    if config.inelastic == "antinodal":
        return np.cos(2.0 * phi) ** 2  # peak-normalised (Fig. 9e1); nu not used
    if config.inelastic == "isotropic":
        return np.ones_like(phi)  # Fig. 9c1
    return np.zeros_like(phi)  # "none"


def inelastic_weight(phi, config: SfConfig = _DEFAULT_CONFIG) -> np.ndarray:
    """W(phi) of the inelastic term by form.

    ``afm_boundary`` is the paper's Eq. 1, the sum of two sin^nu branches with
    eight peaks at the AFM-boundary hot spots; ``antinodal`` cos^2(2 phi);
    ``isotropic`` 1; ``none`` 0.  The alternatives are peak-normalised as
    plotted in Fig. 9e1 / 9c1, so they also change the inelastic strength.
    """
    _validate(config)
    return _inelastic_weight(_finite_array("phi", phi), config)


def _hotspot_profile(phi: np.ndarray, config: SfConfig) -> np.ndarray:
    out = np.zeros_like(phi)
    for centre in _eight_angles(math.radians(config.phi_hs_deg)):
        d = _wrap(phi - centre)
        out = out + np.exp(-((d / config.hotspot_width) ** 2))
    return out


def _rates(phi: np.ndarray, temperatures: np.ndarray, config: SfConfig) -> np.ndarray:
    """Gamma(phi; T) in ps^-1, shape (len(temperatures),) + phi.shape."""
    t = temperatures.reshape((-1,) + (1,) * phi.ndim)
    amp = g_of_x(config.doping) * config.alpha1 * t + config.alpha2 * t**2
    rates = config.gamma_imp + amp * _inelastic_weight(phi, config)
    if config.hotspots and config.hotspot_peak != 0.0:
        rates = rates + config.hotspot_peak * _hotspot_profile(phi, config)
    return rates


def scattering_rate(phi, temperature: float, config: SfConfig = _DEFAULT_CONFIG) -> np.ndarray:
    """Gamma(phi; T, x) = Gamma_imp + Gamma_HS(phi) + [g(x) alpha1 T + alpha2 T^2] W(phi), ps^-1.

    Gamma_HS = H sum_n exp(-(wrap(phi - phi_n) / w)^2) when ``hotspots`` is on
    (effective height H = 31.4 ps^-1 and 1/e half-width w = 0.05 rad by default).
    """
    _validate(config)
    phi = _finite_array("phi", phi)
    return _rates(phi, _temperature(temperature), config)[0]


def cvc_partner_angle(phi, config: SfConfig = _DEFAULT_CONFIG) -> np.ndarray:
    """Angle of the Q = (pi, pi) partner of the FS point at phi ("geometric").

    phi' = atan2(kF sin phi - (pi/a) sgn(sin phi), kF cos phi - (pi/a) sgn(cos phi))
    mod 2 pi: k + Q about the nearest pocket centre.  sgn(0) = 0, so
    cvc_partner_angle(0) = pi exactly.
    """
    _validate(config)
    return _partner_angle(_finite_array("phi", phi), config)


def _partner_angle(phi: np.ndarray, config: SfConfig) -> np.ndarray:
    kf = _fs_at(phi, config)[0]
    q = np.pi / (A_LCCO * ANGSTROM_TO_M)
    px = kf * np.cos(phi) - q * np.sign(np.cos(phi))
    py = kf * np.sin(phi) - q * np.sign(np.sin(phi))
    return np.mod(np.arctan2(py, px), 2.0 * np.pi)


def _epsilon_profile(phi: np.ndarray, config: SfConfig, centres: np.ndarray) -> np.ndarray:
    out = np.zeros_like(phi)
    for centre in centres:
        d = _wrap(phi - centre)
        out = out + np.exp(-0.5 * (d / config.cvc_sigma) ** 2)
    return config.cvc_epsilon * out


def cvc_epsilon_profile(phi, config: SfConfig = _DEFAULT_CONFIG, centres=None) -> np.ndarray:
    """eps(phi) = eps0 sum_n exp(-1/2 (wrap(phi - phi_c,n) / sigma_eps)^2) (Eq. 13).

    ``centres`` are ``cvc_centres(config)``; callers that evaluate eps more than
    once compute them once and pass them in (None computes them here and may
    raise ValueError when the FS misses the zone boundary).
    """
    _validate(config)
    phi = _finite_array("phi", phi)
    if centres is None:
        centres = cvc_centres(config)
    centres = _finite_array("centres", centres)
    if centres.shape != (8,):
        raise ValueError("centres must hold the eight CVC centres")
    return _epsilon_profile(phi, config, centres)


def _cvc_vertex(
    fs: SfFermiSurface, config: SfConfig, centres: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    eps = _epsilon_profile(fs.phi, config, centres)
    _, _, _, pvx, pvy = _fs_at(_partner_angle(fs.phi, config), config)
    return fs.vx + eps * pvx, fs.vy + eps * pvy


def cvc_velocity(config: SfConfig = _DEFAULT_CONFIG) -> tuple[np.ndarray, np.ndarray]:
    """Dressed current J* = v(phi) + eps(phi) v(phi') on the grid, m/s: (J*_x, J*_y).

    Eq. 12 without the 1/(1 - eps^2) prefactor, which the paper drops (p. 19).
    """
    _validate(config)
    return _cvc_vertex(_fermi_surface(config), config, cvc_centres(config))


def _sigma_prefactor(config: SfConfig) -> float:
    """e^2 / (2 pi^2 hbar d), with d = layer_spacing."""
    return ELEMENTARY_CHARGE_C**2 / (
        2.0 * np.pi**2 * HBAR_J_S * config.layer_spacing * ANGSTROM_TO_M
    )


def _spectral_derivative(f: np.ndarray) -> np.ndarray:
    """d/dphi along the last axis of a periodic, uniformly sampled array."""
    n = f.shape[-1]
    k = np.arange(n // 2 + 1, dtype=float)
    if n % 2 == 0:
        k[-1] = 0.0
    return np.fft.irfft(np.fft.rfft(f, axis=-1) * (1j * k), n=n, axis=-1)


def _exp_moments(r0: np.ndarray) -> np.ndarray:
    """E_n(R) = int_0^1 t^n exp(-R t) dt for n = 0..6, shape (7,) + R.shape.

    Below R = 2 the 28-term series sum_k (-R)^k / (k! (n + k + 1)) (k outer,
    n inner); from R = 2 up the recursion e_0 = -expm1(-R)/R,
    e_n = (n e_{n-1} - exp(-R)) / R.  Each branch sees only its own elements.
    """
    out = np.empty((_N_MOMENTS,) + r0.shape)
    small = r0 < _SERIES_SWITCH
    rs = r0[small]
    neg = -rs
    term = np.ones_like(rs)
    acc = np.zeros((_N_MOMENTS, rs.size))
    scratch = np.empty_like(rs)
    for k in range(_SERIES_TERMS):
        for n in range(_N_MOMENTS):
            np.divide(term, n + k + 1, out=scratch)
            acc[n] += scratch
        np.multiply(term, neg, out=term)
        np.divide(term, k + 1, out=term)
    out[:, small] = acc
    large = ~small
    rl = r0[large]
    em = np.exp(-rl)
    e = -np.expm1(-rl) / rl
    for n in range(_N_MOMENTS):
        if n > 0:
            e = (n * e - em) / rl
        out[n][large] = e
    return out


def _close_orbits(cell: np.ndarray, decay: np.ndarray, total: np.ndarray) -> np.ndarray:
    """Solve M_s = I_s + A_s M_{s+1} (indices mod N) for every s: the two-pass closure.

    cell: (nb, nB, 2, N) per-cell integrals I; decay: (nb, nB, N) cell survival A;
    total: (nb, nB) h sum_s r_s.  The forward pass accumulates sum_s (prod_{l<s} A_l) I_s
    in scan order, M_0 = that / (1 - exp(-total)), then the recurrence runs back from
    s = N - 1 to 1.  Every (batch, field, component) closes in the same pass.
    """
    i_scan = np.ascontiguousarray(np.moveaxis(cell, -1, 0))  # (N, nb, nB, 2)
    a_scan = np.ascontiguousarray(np.moveaxis(decay, -1, 0))[..., None]  # (N, nb, nB, 1)
    n = i_scan.shape[0]
    survival = np.empty_like(a_scan)
    survival[0] = 1.0
    np.cumprod(a_scan[:-1], axis=0, out=survival[1:])
    m = np.empty_like(i_scan)
    m[0] = np.sum(survival * i_scan, axis=0) / (-np.expm1(-total))[..., None]
    np.multiply(a_scan[n - 1], m[0], out=m[n - 1])
    m[n - 1] += i_scan[n - 1]
    for s in range(n - 2, 0, -1):
        np.multiply(a_scan[s], m[s + 1], out=m[s])
        m[s] += i_scan[s]
    return np.ascontiguousarray(np.moveaxis(m, 0, -1))


def _orbit_solution(f: np.ndarray, r: np.ndarray, h: float) -> np.ndarray:
    """L = J/Gamma + M on forward-ordered past orbits (the exact per-cell integrator).

    f: (nb, 2, N) J/Gamma in scan order; r: (nb, nB, N) Gamma / (omega |B|) in scan
    order.  J/Gamma is a cubic Hermite in each cell (slopes from the spectral
    derivative of the reordered f, taken once per batch row and field sign), Gamma/omega
    is linear, and the cell integral is exact in the E_n moments.  Returns (nb, nB, 2, N).
    """
    fp = _spectral_derivative(f)
    r1 = np.roll(r, -1, axis=-1)
    k = 0.5 * (r1 - r) * h
    decay = np.exp(-0.5 * (r + r1) * h)
    e = _exp_moments(r * h)
    f1 = np.roll(f, -1, axis=-1)
    fp1 = np.roll(fp, -1, axis=-1)
    slope = (f1 - f) / h
    c0 = fp
    c1 = 6.0 * slope - 4.0 * fp - 2.0 * fp1
    c2 = 3.0 * fp + 3.0 * fp1 - 6.0 * slope
    k2 = 0.5 * k * k
    q0 = e[0] - k * e[2] + k2 * e[4]
    q1 = e[1] - k * e[3] + k2 * e[5]
    q2 = e[2] - k * e[4] + k2 * e[6]
    cell = h * (
        c0[:, None] * q0[:, :, None] + c1[:, None] * q1[:, :, None] + c2[:, None] * q2[:, :, None]
    )
    return f[:, None] + _close_orbits(cell, decay, h * np.sum(r, axis=-1))


def _contract(prefactor: float, h: float, weighted: np.ndarray, lvec: np.ndarray) -> np.ndarray:
    """prefactor h sum_m (w V_i)_m L_j,m for i, j in (x, y); lvec (..., 2, N) -> (..., 2, 2)."""
    out = np.empty(lvec.shape[:-2] + (2, 2))
    for i in range(2):
        for j in range(2):
            out[..., i, j] = prefactor * h * np.sum(weighted[i] * lvec[..., j, :], axis=-1)
    return out


def _conductivity_batch(
    config: SfConfig, fs: SfFermiSurface, rates: np.ndarray, fields: np.ndarray
) -> np.ndarray:
    """sigma_ij(B) in S/m for a batch of scattering rates, shape (nb, nB, 2, 2).

    rates: (nb, N) in s^-1 on fs.phi (one row per temperature or doping).  Bare
    vertices, V = J = v (no CVCs in rho or MR).  The past orbit runs to increasing phi
    for B > 0 (holes) and in the order (0, N-1, ..., 1) for B < 0; B = 0 is the closed
    form L = J/Gamma.  Vectorised over (batch, field, component).
    """
    n = config.n_phi
    h = 2.0 * np.pi / n
    prefactor = _sigma_prefactor(config)
    velocity = np.stack([fs.vx, fs.vy])
    weighted = fs.weight * velocity
    f = velocity / rates[:, None, :]
    sigma = np.empty((rates.shape[0], fields.size, 2, 2))
    zero = fields == 0.0
    if zero.any():
        sigma[:, zero] = _contract(prefactor, h, weighted, f)[:, None]
    for sign in (1.0, -1.0):
        picked = np.sign(fields) == sign
        if not picked.any():
            continue
        order = np.arange(n) if sign > 0 else (-np.arange(n)) % n
        speed = fs.omega_per_tesla[order] * np.abs(fields[picked])[:, None]  # (nB, N)
        lvec = _orbit_solution(f[..., order], rates[:, None, order] / speed, h)
        if sign < 0:
            unordered = np.empty_like(lvec)
            unordered[..., order] = lvec
            lvec = unordered
        sigma[:, picked] = _contract(prefactor, h, weighted, lvec)
    return sigma


def _resistivity(sigma: np.ndarray) -> np.ndarray:
    """(rho_xx, rho_yx) in micro-Ohm cm from sigma (..., 2, 2) by the explicit 2x2 inverse."""
    det = sigma[..., 0, 0] * sigma[..., 1, 1] - sigma[..., 0, 1] * sigma[..., 1, 0]
    rho_xx = sigma[..., 1, 1] / det * _OHM_M_TO_MUOHM_CM
    rho_yx = -sigma[..., 1, 0] / det * _OHM_M_TO_MUOHM_CM
    return np.stack([rho_xx, rho_yx], axis=-1)


def conductivity_tensor(config: SfConfig, temperature: float, fields) -> np.ndarray:
    """Shockley-Chambers sigma_ij(B) in S/m, shape (n_fields, 2, 2), bare vertices.

    sigma_ij = e^2/(2 pi^2 hbar d) oint dphi w V_i L_j with L_j the past-orbit
    integral of J_j exp(-int Gamma dt), solved exactly cell by cell.
    """
    _validate(config)
    fs = _fermi_surface(config)
    rates = _rates(fs.phi, _temperature(temperature), config) * _PS_INV
    return _conductivity_batch(config, fs, rates, _field_array(fields))[0]


def resistivity_tensor(config: SfConfig, temperature: float, fields) -> np.ndarray:
    """(rho_xx, rho_yx) in micro-Ohm cm at each field, shape (n_fields, 2)."""
    return _resistivity(conductivity_tensor(config, temperature, fields))


def zero_field_resistivity(config: SfConfig, temperatures) -> np.ndarray:
    """rho(0, T) in micro-Ohm cm from the closed form sigma_xx = pref h sum w vx^2 / Gamma."""
    _validate(config)
    t = _temperature_array(temperatures)
    fs = _fermi_surface(config)
    rates = _rates(fs.phi, t, config) * _PS_INV
    h = 2.0 * np.pi / config.n_phi
    sxx = _sigma_prefactor(config) * h * np.sum(fs.weight * fs.vx * fs.vx / rates, axis=-1)
    return _OHM_M_TO_MUOHM_CM / sxx


def field_grid(field_max: float = SF_FIELD_MAX, step: float = SF_FIELD_STEP) -> np.ndarray:
    """0 .. field_max in ``step`` increments (T), inclusive; field_max snaps down."""
    field_max = _positive("field_max", field_max)
    step = _positive("step", step)
    n = int(math.floor(field_max / step + 1e-9))
    return step * np.arange(n + 1)


def field_derivative(rho, step: float) -> np.ndarray:
    """d rho / d mu0 H along the last axis: central inside, second-order one-sided ends.

    Equals ``np.gradient(rho, step, axis=-1, edge_order=2)``, written out so the Rust
    twin can match it term by term.
    """
    step = _positive("step", step)
    rho = _finite_array("rho", rho)
    if rho.ndim == 0 or rho.shape[-1] < 3:
        raise ValueError("rho needs at least three points along the last axis")
    out = np.empty_like(rho)
    out[..., 1:-1] = (rho[..., 2:] - rho[..., :-2]) / (2.0 * step)
    out[..., 0] = (-3.0 * rho[..., 0] + 4.0 * rho[..., 1] - rho[..., 2]) / (2.0 * step)
    out[..., -1] = (3.0 * rho[..., -1] - 4.0 * rho[..., -2] + rho[..., -3]) / (2.0 * step)
    return out


def magnetoresistance(
    config: SfConfig,
    temperatures=SF_MR_TEMPERATURES,
    field_max: float = SF_FIELD_MAX,
    field_step: float = SF_FIELD_STEP,
) -> SfMagnetoresistance:
    """rho(H, T), MR and d rho / d mu0 H on the field grid for each temperature.

    One vectorised solve over every (T, H) pair.  h_over_t is H / T (inf or NaN at
    T = 0).
    """
    _validate(config)
    temps = _temperature_array(temperatures)
    fields = field_grid(field_max, field_step)
    if fields.size < 3:
        raise ValueError("field_max must span at least two field steps")
    fs = _fermi_surface(config)
    rates = _rates(fs.phi, temps, config) * _PS_INV
    rho = _resistivity(_conductivity_batch(config, fs, rates, fields))
    rho_xx = np.ascontiguousarray(rho[..., 0])
    rho_yx = np.ascontiguousarray(rho[..., 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        h_over_t = fields[None, :] / temps[:, None]
    return SfMagnetoresistance(
        temperatures=temps,
        fields=fields,
        rho_xx=rho_xx,
        rho_yx=rho_yx,
        mr=rho_xx / rho_xx[:, :1] - 1.0,
        drho_dh=field_derivative(rho_xx, field_step),
        h_over_t=h_over_t,
    )


def high_field_slope(
    config: SfConfig,
    temperature: float,
    field: float = SF_SLOPE_FIELD,
    step: float = SF_FIELD_STEP,
) -> float:
    """gamma_1 = d rho / d mu0 H at ``field`` (micro-Ohm cm / T): (rho(H + h) - rho(H - h)) / 2h."""
    field = _real("field", field)
    step = _positive("step", step)
    rho = resistivity_tensor(config, temperature, [field - step, field + step])[:, 0]
    return float((rho[1] - rho[0]) / (2.0 * step))


def _det3(m: list[list[float]]) -> float:
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def fit_resistivity_polynomial(temperatures, rho) -> SfResistivityFit:
    """Least-squares rho0 + A1 T + A2 T^2 by Cramer's rule on the 3x3 normal equations.

    The power sums are accumulated sequentially, as in the Rust twin.
    """
    t = _finite_array("temperatures", temperatures)
    y = _finite_array("rho", rho)
    if t.ndim != 1 or t.shape != y.shape:
        raise ValueError("temperatures and rho must be 1D arrays of equal length")
    if np.unique(t).size < 3:
        raise ValueError("the fit needs at least three distinct temperatures")
    s = [0.0] * 5
    r = [0.0] * 3
    for tk, v in zip(t.tolist(), y.tolist(), strict=True):
        p = 1.0
        for k in range(5):
            s[k] += p
            if k < 3:
                r[k] += v * p
            p *= tk
    normal = [[s[0], s[1], s[2]], [s[1], s[2], s[3]], [s[2], s[3], s[4]]]
    det = _det3(normal)
    coeffs = []
    for c in range(3):
        mc = [row[:] for row in normal]
        for row in range(3):
            mc[row][c] = r[row]
        coeffs.append(_det3(mc) / det)
    return SfResistivityFit(
        rho0=coeffs[0], a1=coeffs[1], a2=coeffs[2], t_min=float(t.min()), t_max=float(t.max())
    )


def _fit_temperatures(t_range, step) -> np.ndarray:
    t_min = _non_negative("t_range[0]", t_range[0])
    t_max = _real("t_range[1]", t_range[1])
    step = _positive("step", step)
    if not t_max > t_min:
        raise ValueError("t_range must be increasing")
    return t_min + step * np.arange(int(math.floor((t_max - t_min) / step + 1e-9)) + 1)


def resistivity_fit(
    config: SfConfig, t_range=SF_RHO_FIT_RANGE, step: float = 1.0
) -> SfResistivityFit:
    """rho0 + A1 T + A2 T^2 fitted to the model's rho(0, T) on t_range (15-60 K, Fig. 6)."""
    temps = _fit_temperatures(t_range, step)
    return fit_resistivity_polynomial(temps, zero_field_resistivity(config, temps))


def hall_coefficient_curve(config: SfConfig, temperatures, cvc: bool = True) -> np.ndarray:
    """Jones-Zener R_H(T) in m^3/C (Eqs. 14-15), with or without current-vertex corrections.

    sigma_xx = pref sum w v_x J*_x / Gamma (one dressed vertex), sigma_xy / B =
    pref (e/hbar) sum (J*_x / Gamma) d(J*_y / Gamma)/dphi (both dressed, hole sign),
    R_H = (sigma_xy / B) / (sigma_xx sigma_yy).  cvc=False uses J* = v.
    """
    _validate(config)
    temps = _temperature_array(temperatures)
    fs = _fermi_surface(config)
    rates = _rates(fs.phi, temps, config) * _PS_INV
    if cvc:
        jx, jy = _cvc_vertex(fs, config, cvc_centres(config))
    else:
        jx, jy = fs.vx, fs.vy
    h = 2.0 * np.pi / config.n_phi
    prefactor = _sigma_prefactor(config)
    sxx = prefactor * h * np.sum(fs.weight * fs.vx * jx / rates, axis=-1)
    syy = prefactor * h * np.sum(fs.weight * fs.vy * jy / rates, axis=-1)
    sxy_b = (
        prefactor
        * (ELEMENTARY_CHARGE_C / HBAR_J_S)
        * h
        * np.sum(jx / rates * _spectral_derivative(jy / rates), axis=-1)
    )
    return sxy_b / (sxx * syy)


def hall_coefficient(config: SfConfig, temperature: float, cvc: bool = True) -> float:
    """Jones-Zener R_H at one temperature, m^3/C (see :func:`hall_coefficient_curve`)."""
    return float(hall_coefficient_curve(config, _temperature(temperature), cvc)[0])


def hall_coefficient_chambers(config: SfConfig, temperature: float, field: float = 0.1) -> float:
    """rho_yx / B from the Chambers solution (no CVC), m^3/C: the low-field check of JZ."""
    b = _real("field", field)
    if b == 0.0:
        raise ValueError("field must be nonzero")
    rho = resistivity_tensor(config, temperature, [b])
    return float(rho[0, 1] / _OHM_M_TO_MUOHM_CM / b)


def drude_hall_coefficient(config: SfConfig = _DEFAULT_CONFIG) -> float:
    """1 / (n e) with n = luttinger_holes / (a^2 d): the Drude R_H of the fixed x = 0.159 FS."""
    n = luttinger_holes(config) / (
        (A_LCCO * ANGSTROM_TO_M) ** 2 * config.layer_spacing * ANGSTROM_TO_M
    )
    return 1.0 / (n * ELEMENTARY_CHARGE_C)


def drude_hall_coefficient_nominal(config: SfConfig = _DEFAULT_CONFIG) -> float:
    """a^2 d / ((1 - x) e), m^3/C: the Drude R_H for the nominal 1 - x holes per Cu."""
    _validate(config)
    n = (1.0 - config.doping) / (
        (A_LCCO * ANGSTROM_TO_M) ** 2 * config.layer_spacing * ANGSTROM_TO_M
    )
    return 1.0 / (n * ELEMENTARY_CHARGE_C)


def hall_ratio_to_case_a(config: SfConfig, temperature: float) -> float:
    """R_H(config, T, CVC) / R_H(case (a), CVC): (d)/(a) of Fig. 13 for the defaults.

    Case (a) is the Gamma_0-only Fermi surface (``inelastic="none"``, no hot
    spots).  A uniform Gamma cancels in Jones-Zener, so the denominator does not
    depend on T (5.309e-10 m^3/C at the defaults).  The paper's 4.3/8.7 = 0.49.
    """
    num = hall_coefficient(config, temperature, cvc=True)
    den = hall_coefficient(replace(config, inelastic="none", hotspots=False), temperature, cvc=True)
    return num / den


def _dos_average(values: np.ndarray, fs: SfFermiSurface) -> float:
    return float(np.sum(values * fs.weight) / np.sum(fs.weight))


def _lambda_tr(config: SfConfig, average: float) -> float:
    return (
        HBAR_J_S
        * g_of_x(config.doping)
        * config.alpha1
        * _PS_INV
        * average
        / (2.0 * math.pi * KB_J_K)
    )


def lambda_tr(config: SfConfig = _DEFAULT_CONFIG) -> float:
    """lambda_tr = hbar g(x) alpha1 <W>_FS / (2 pi kB), with <W>_FS DOS-weighted.

    lambda_tr, the Fermi-surface-averaged T-linear coupling (Allen's form applied to
    the microscopic rate).  Only the T-linear part of the inelastic term enters;
    0.4176 g for the paper's form.  Allen's hbar Gamma = 2 pi lambda kB T is a phonon
    result, and this is not Allen's resistivity-derived lambda: that reading is
    :func:`lambda_resistive`.
    """
    _validate(config)
    fs = _fermi_surface(config)
    return _lambda_tr(config, _dos_average(_inelastic_weight(fs.phi, config), fs))


@lru_cache(maxsize=1)
def _lambda_tr_unit() -> float:
    """lambda_tr per unit g for the paper's form (the calibration coefficient)."""
    return lambda_tr(_DEFAULT_CONFIG)


def _drude_resistivity_per_rate(fs: SfFermiSurface, config: SfConfig) -> float:
    h = 2.0 * np.pi / config.n_phi
    return 1.0 / (_sigma_prefactor(config) * h * float(np.sum(fs.weight * fs.vx * fs.vx)))


def drude_resistivity_per_rate(config: SfConfig = _DEFAULT_CONFIG) -> float:
    """rho_D in Ohm m s: the model's zero-field rho per unit isotropic rate on the same FS.

    rho = rho_D Gamma exactly when Gamma is uniform.
    """
    _validate(config)
    return _drude_resistivity_per_rate(_fermi_surface(config), config)


def _lambda_rho(a1: float, rho_d: float) -> float:
    return HBAR_J_S * (a1 / _OHM_M_TO_MUOHM_CM) / (2.0 * math.pi * KB_J_K * rho_d)


def lambda_resistive(config: SfConfig = _DEFAULT_CONFIG) -> float:
    """lambda_rho, the Allen-Drude coupling hbar A1 / (2 pi kB rho_D) from the model's own rho(T).

    A1 is the linear coefficient of :func:`resistivity_fit` (15-60 K).  For a uniform
    Gamma it equals lambda_tr of the isotropic form; with hot spots the cold regions
    short-circuit the resistivity, so lambda_rho spans a compressed range across the dome.
    """
    a1 = resistivity_fit(config).a1
    return _lambda_rho(a1, drude_resistivity_per_rate(config))


def doping_sweep(
    config: SfConfig = _DEFAULT_CONFIG,
    temperature: float = SF_FIT_TEMPERATURE,
    field: float = SF_FIT_FIELD,
) -> SfDopingSweep:
    """g, Tc, lambda_tr, lambda_rho, rho(0, T) and MR(field, T) at the 19 tabulated dopings.

    ``config.doping`` is ignored; every other field applies.  The Fermi surface is
    the x = 0.159 one at every x: only g(x) varies (p. 6).
    """
    _validate(config)
    t = _non_negative("temperature", temperature)
    field = _real("field", field)
    fs = _fermi_surface(config)
    configs = [replace(config, doping=x) for x in DOPINGS_LCCO]
    rates = np.stack([_rates(fs.phi, np.array([t]), c)[0] for c in configs]) * _PS_INV
    rho = _resistivity(_conductivity_batch(config, fs, rates, np.array([0.0, field])))[..., 0]
    average = _dos_average(_inelastic_weight(fs.phi, config), fs)
    fit_temps = _fit_temperatures(SF_RHO_FIT_RANGE, 1.0)
    fit_rates = np.stack([_rates(fs.phi, fit_temps, c) for c in configs]) * _PS_INV
    h = 2.0 * np.pi / config.n_phi
    sxx = _sigma_prefactor(config) * h * np.sum(fs.weight * fs.vx * fs.vx / fit_rates, axis=-1)
    fit_rho = _OHM_M_TO_MUOHM_CM / sxx
    rho_d = _drude_resistivity_per_rate(fs, config)
    lambda_rho = [
        _lambda_rho(fit_resistivity_polynomial(fit_temps, row).a1, rho_d) for row in fit_rho
    ]
    return SfDopingSweep(
        dopings=np.array(DOPINGS_LCCO),
        g=np.array([g_of_x(x) for x in DOPINGS_LCCO]),
        tc=np.array([tc_of_x(x) for x in DOPINGS_LCCO]),
        lambda_tr=np.array([_lambda_tr(c, average) for c in configs]),
        lambda_rho=np.array(lambda_rho),
        rho0=rho[:, 0],
        mr=rho[:, 1] / rho[:, 0] - 1.0,
        temperature=t,
        field=field,
    )


def equivalent_doping(lam: float) -> float | None:
    """LCCO doping whose lambda_tr (paper's form) equals ``lam``, by inverse interpolation.

    g = lam / lambda_tr(g = 1); the first table segment (ascending x) with
    g1 <= g <= g0 gives x0 + (x1 - x0)(g0 - g)/(g0 - g1) (x0 on a flat segment).
    None outside [lambda_tr(0.188), lambda_tr(0.116)] or for non-finite input.
    """
    try:
        lam = float(lam)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(lam):
        return None
    g = lam / _lambda_tr_unit()
    for (x0, g0), (x1, g1) in zip(G_TABLE_LCCO[:-1], G_TABLE_LCCO[1:], strict=True):
        if g1 <= g <= g0:
            return x0 if g0 == g1 else x0 + (x1 - x0) * (g0 - g) / (g0 - g1)
    return None


# ---------------------------------------------------------------------------
# Part B — pairing map (SPECULATIVE)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SfPairingMap:
    """Tc(lambda) map (SPECULATIVE): the empirical power law or McMillan's form, best fit."""

    form: str = "power_law"  # SF_PAIRING_FORMS
    lambda_c: float = SF_POWER_LAW_LAMBDA_C  # power-law threshold
    exponent: float = SF_POWER_LAW_EXPONENT  # power-law p
    lambda_ref: float = SF_POWER_LAW_LAMBDA_REF
    tc_ref: float = SF_POWER_LAW_TC_REF  # K
    omega_k: float = SF_MCMILLAN_OMEGA_K  # K, McMillan prefactor scale
    mu_star: float = SF_MCMILLAN_MU_STAR


_DEFAULT_PAIRING = SfPairingMap()


def _validate_pairing(pairing: SfPairingMap) -> None:
    if pairing.form not in SF_PAIRING_FORMS:
        raise ValueError(f"pairing form must be one of {SF_PAIRING_FORMS}")
    lambda_c = _non_negative("lambda_c", pairing.lambda_c)
    if not _real("lambda_ref", pairing.lambda_ref) > lambda_c:
        raise ValueError("lambda_ref must exceed lambda_c")
    _positive("exponent", pairing.exponent)
    _positive("tc_ref", pairing.tc_ref)
    _positive("omega_k", pairing.omega_k)
    if not 0.0 <= _real("mu_star", pairing.mu_star) < 1.0 / 0.62:
        raise ValueError("mu_star must lie in [0, 1/0.62)")


def _tc_from_lambda(lam: np.ndarray, pairing: SfPairingMap) -> np.ndarray:
    if pairing.form == "power_law":
        x = np.clip((lam - pairing.lambda_c) / (pairing.lambda_ref - pairing.lambda_c), 0.0, None)
        return pairing.tc_ref * x**pairing.exponent
    den = lam - pairing.mu_star * (1.0 + 0.62 * lam)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        tc = (pairing.omega_k / 1.45) * np.exp(-1.04 * (1.0 + lam) / den)
    return np.where(den > 0, tc, 0.0)


def tc_from_lambda(lam, pairing: SfPairingMap = _DEFAULT_PAIRING) -> float | np.ndarray:
    """Tc(lambda) in K (SPECULATIVE); 0 at and below the threshold.

    power_law: T_ref ((lambda - lambda_c)/(lambda_ref - lambda_c))^p, an EMPIRICAL
    interpolation of the three superconducting Fig. 5c labels.  mcmillan:
    (omega/1.45) exp[-1.04 (1 + lambda) / (lambda - mu* (1 + 0.62 lambda))], the phonon
    form with theta_D -> omega, shown failing.
    """
    _validate_pairing(pairing)
    return _scalar_or_array(_tc_from_lambda(_finite_array("lambda", lam), pairing))


def tc_ceiling(pairing: SfPairingMap = _DEFAULT_PAIRING) -> float:
    """Largest Tc the map reaches, K (SPECULATIVE): inf for the power law; for McMillan
    (omega/1.45) exp(-1.04 / (1 - 0.62 mu*)).
    """
    _validate_pairing(pairing)
    if pairing.form == "power_law":
        return math.inf
    return (pairing.omega_k / 1.45) * math.exp(-1.04 / (1.0 - 0.62 * pairing.mu_star))


def lambda_threshold(pairing: SfPairingMap = _DEFAULT_PAIRING) -> float:
    """lambda at and below which Tc = 0 (SPECULATIVE): lambda_c, or mu*/(1 - 0.62 mu*)
    for McMillan.
    """
    _validate_pairing(pairing)
    if pairing.form == "power_law":
        return pairing.lambda_c
    return pairing.mu_star / (1.0 - 0.62 * pairing.mu_star)


def lambda_from_tc(tc: float, pairing: SfPairingMap = _DEFAULT_PAIRING) -> float:
    """Inverse of :func:`tc_from_lambda` (SPECULATIVE); ValueError unless 0 < Tc < ceiling.

    Also ValueError where the inverse rounds onto the threshold, so Tc(lambda) = 0
    (default power law: Tc <= 6.173e-6 K); the glue quantities divide by Tc(lambda_bar).
    """
    tc = _real("tc", tc)
    if not 0.0 < tc < tc_ceiling(pairing):
        raise ValueError("tc must lie in (0, tc_ceiling)")
    if pairing.form == "power_law":
        lam = pairing.lambda_c + (pairing.lambda_ref - pairing.lambda_c) * (
            tc / pairing.tc_ref
        ) ** (1.0 / pairing.exponent)
    else:
        big_l = -math.log(1.45 * tc / pairing.omega_k) / 1.04
        lam = (1.0 + big_l * pairing.mu_star) / (big_l * (1.0 - 0.62 * pairing.mu_star) - 1.0)
    if not _tc_from_lambda(np.asarray(lam), pairing) > 0.0:
        raise ValueError("tc lies below the pairing map's resolvable range: Tc(lambda) = 0")
    return lam


def log_sensitivity(lam, pairing: SfPairingMap = _DEFAULT_PAIRING) -> float | np.ndarray:
    """S = d ln Tc / d ln lambda (SPECULATIVE), above the threshold.

    power_law: p lambda / (lambda - lambda_c); mcmillan: 1.04 lambda (1 + 0.38 mu*) / D^2
    with D = lambda - mu* (1 + 0.62 lambda).
    """
    _validate_pairing(pairing)
    lam = _finite_array("lambda", lam)
    if pairing.form == "power_law":
        return _scalar_or_array(pairing.exponent * lam / (lam - pairing.lambda_c))
    den = lam - pairing.mu_star * (1.0 + 0.62 * lam)
    return _scalar_or_array(1.04 * lam * (1.0 + 0.38 * pairing.mu_star) / den**2)


def calibration_points() -> tuple[tuple[float, float, float], ...]:
    """(x, Tc, lambda_tr) at the Fig. 5c labels, with lambda_tr = lambda_tr(g = 1) g(x)
    as in the calibration (SPECULATIVE map input)."""
    unit = _lambda_tr_unit()
    return tuple((x, tc, unit * g_of_x(x)) for x, tc in TC_LABELS_FIG5C)


def pairing_fit_rms(pairing: SfPairingMap = _DEFAULT_PAIRING) -> float:
    """rms Tc(lambda_tr) - Tc over the four Fig. 5c labels, K (SPECULATIVE)."""
    points = calibration_points()
    sq = [(float(tc_from_lambda(lam, pairing)) - tc) ** 2 for _x, tc, lam in points]
    return math.sqrt(sum(sq) / len(points))


# ---------------------------------------------------------------------------
# Part C — moire glue what-if (SPECULATIVE)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SfGlueConfig:
    """Moire glue-modulation what-if (SPECULATIVE)."""

    eta: float = SF_GLUE_ETA  # depth of lambda(r), [0, 1)
    tc_target: float = TC_FETE_FILM  # K, operating Tc*, (0, tc_ceiling)
    delta_avg: float = DELTA_AVG  # meV: the film's mean gap AT TC_FETE_FILM
    coherence: str = SF_GLUE_COHERENCE  # SF_COHERENCE_MODELS
    moire_period: float = SF_GLUE_MOIRE_PERIOD  # L_m, Angstrom, finite > 0
    pairing: SfPairingMap = SfPairingMap()


_DEFAULT_GLUE = SfGlueConfig()


@dataclass(frozen=True)
class SfGlueResult:
    """The glue-modulated gap map and its imprint (SPECULATIVE); maps have the pattern's shape."""

    lambda_bar: float
    sensitivity: float  # S(lambda_bar)
    coherence_length: float  # xi(Tc*), Angstrom (the clean value in the local limit)
    coherence_factor: float  # F = exp(-xi / L_m); 1.0 in the local limit
    moire_period: float  # L_m, Angstrom
    lambda_map: np.ndarray
    tc_map: np.ndarray  # local Tc(lambda(r)), K
    gap: np.ndarray  # dressed Delta(r), meV
    relative_modulation: float  # dressed (max - min) / (2 Delta_bar)
    local_relative_modulation: float
    local_normal_fraction: float  # mean[tc_map == 0]
    normal_fraction: float  # local_normal_fraction if coherence == "local" else 0.0


def _validate_glue(glue: SfGlueConfig) -> None:
    if glue.coherence not in SF_COHERENCE_MODELS:
        raise ValueError(f"coherence must be one of {SF_COHERENCE_MODELS}")
    _positive("moire_period", glue.moire_period)
    if not 0.0 <= _real("eta", glue.eta) < 1.0:
        raise ValueError("eta must lie in [0, 1)")
    _positive("delta_avg", glue.delta_avg)
    _validate_pairing(glue.pairing)
    if not 0.0 < _real("tc_target", glue.tc_target) < tc_ceiling(glue.pairing):
        raise ValueError("tc_target must lie in (0, tc_ceiling)")
    lambda_from_tc(glue.tc_target, glue.pairing)  # raises where Tc(lambda_bar) would be 0


def moire_profile(pattern) -> np.ndarray:
    """m~ = -cos(pi m) for a pattern m in [0, 1] (SPECULATIVE what-if input):
    gap_modulation's own profile (phase pi).
    """
    return -np.cos(np.pi * _finite_array("pattern", pattern))


def coherence_length(glue: SfGlueConfig = _DEFAULT_GLUE) -> float:
    """xi(Tc*) in Angstrom (SPECULATIVE): 20 A (TC_FETE_FILM / Tc*)^s.

    s = 1 for "clean" (and reported for "local"), s = 1/2 for "dirty": textbook
    limits from the repo's BCS estimate DEFAULT_COHERENCE_LENGTH at TC_FETE_FILM.
    """
    _validate_glue(glue)
    ratio = TC_FETE_FILM / glue.tc_target
    if glue.coherence == "dirty":
        return DEFAULT_COHERENCE_LENGTH * math.sqrt(ratio)
    return DEFAULT_COHERENCE_LENGTH * ratio


def coherence_factor(glue: SfGlueConfig = _DEFAULT_GLUE) -> float:
    """F = exp(-xi / L_m), the CPDM dressing (SPECULATIVE); 1.0 in the local limit.

    The formula of superconducting.cpdm_amplitude, evaluated with exp directly:
    cpdm_amplitude clamps xi/L_m at EXPONENT_CLAMP = 100, which a low Tc* or a
    short L_m exceeds.
    """
    if glue.coherence == "local":
        _validate_glue(glue)
        return 1.0
    return math.exp(-(coherence_length(glue) / glue.moire_period))


def glue_gap_scale(glue: SfGlueConfig = _DEFAULT_GLUE) -> float:
    """Delta_bar = delta_avg Tc* / TC_FETE_FILM, meV (SPECULATIVE).

    It keeps the film's 2 Delta / kB Tc at every operating Tc*.
    """
    _validate_glue(glue)
    return glue.delta_avg * (glue.tc_target / TC_FETE_FILM)


def glue_modulated_gap(pattern, glue: SfGlueConfig = _DEFAULT_GLUE) -> SfGlueResult:
    """The gap when the moire modulates the pairing glue instead of the gap (SPECULATIVE).

    lambda(r) = lambda_bar (1 + eta m~(r)), Tc(r) = Tc(lambda(r)), Delta_loc =
    Delta_bar Tc(r) / Tc(lambda_bar), dressed Delta = Delta_bar + F (Delta_loc -
    Delta_bar).  Tc(r) = 0 regions exist only in the local limit (F < 1 keeps
    Delta >= Delta_bar (1 - F) > 0); where they are smaller than xi they are unphysical.
    """
    _validate_glue(glue)
    m = _finite_array("pattern", pattern)
    if m.size == 0:
        raise ValueError("pattern must not be empty")
    pairing = glue.pairing
    lam_bar = lambda_from_tc(glue.tc_target, pairing)
    lam = lam_bar * (1.0 + glue.eta * moire_profile(m))
    tc = _tc_from_lambda(lam, pairing)
    tc_bar = float(_tc_from_lambda(np.asarray(lam_bar), pairing))
    d_bar = glue_gap_scale(glue)
    d_loc = d_bar * tc / tc_bar
    f = coherence_factor(glue)
    gap = d_bar + f * (d_loc - d_bar)
    local_normal = float(np.mean(tc == 0.0))
    return SfGlueResult(
        lambda_bar=lam_bar,
        sensitivity=float(log_sensitivity(lam_bar, pairing)),
        coherence_length=coherence_length(glue),
        coherence_factor=f,
        moire_period=glue.moire_period,
        lambda_map=lam,
        tc_map=tc,
        gap=gap,
        relative_modulation=float((gap.max() - gap.min()) / (2.0 * d_bar)),
        local_relative_modulation=float((d_loc.max() - d_loc.min()) / (2.0 * d_bar)),
        local_normal_fraction=local_normal,
        normal_fraction=local_normal if glue.coherence == "local" else 0.0,
    )


def linearized_glue_amplitude(glue: SfGlueConfig = _DEFAULT_GLUE) -> float:
    """F Delta_bar eta S(lambda_bar), meV (SPECULATIVE): the small-eta dressed amplitude.

    As eta -> 0 the glue map tends to gap_modulation(pattern, Delta_bar, this),
    whose default phase shift pi gives the same -cos(pi m) profile.
    """
    _validate_glue(glue)
    lam_bar = lambda_from_tc(glue.tc_target, glue.pairing)
    return (
        coherence_factor(glue)
        * glue_gap_scale(glue)
        * glue.eta
        * float(log_sensitivity(lam_bar, glue.pairing))
    )


def bcs_route_relative_modulation() -> float:
    """Amplitude-route reference DELTA_AMPLITUDE / DELTA_AVG = 0.165 (SPECULATIVE comparison).

    "amplitude route, measured gaps (13.5 K)": the measured gaps are already
    coherence-dressed, so this is a constant, not a curve in Tc*.
    """
    return DELTA_AMPLITUDE / DELTA_AVG


def local_imprint(
    eta: float, tc_target: float, pairing: SfPairingMap = _DEFAULT_PAIRING
) -> tuple[float, float]:
    """(local exact, local linear) imprints for a pattern spanning [0, 1] (SPECULATIVE).

    exact = [Tc(lambda_bar (1 + eta)) - Tc(lambda_bar (1 - eta))] / (2 Tc(lambda_bar)),
    linear = eta S(lambda_bar); both independent of L_m.
    """
    eta = _real("eta", eta)
    if not 0.0 <= eta < 1.0:
        raise ValueError("eta must lie in [0, 1)")
    lam_bar = lambda_from_tc(tc_target, pairing)
    tc_bar = float(tc_from_lambda(lam_bar, pairing))
    exact = (
        float(tc_from_lambda(lam_bar * (1 + eta), pairing))
        - float(tc_from_lambda(lam_bar * (1 - eta), pairing))
    ) / (2 * tc_bar)
    return exact, eta * float(log_sensitivity(lam_bar, pairing))


def imprint_curve(glue: SfGlueConfig, tc_targets) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(dressed exact, local exact, local linear) against the operating Tc* (SPECULATIVE).

    Uses glue.coherence and glue.moire_period; dressed = F(Tc*) local exact.
    """
    _validate_glue(glue)
    targets = _finite_array("tc_targets", tc_targets)
    if targets.ndim != 1:
        raise ValueError("tc_targets must be a 1D sequence")
    dressed = np.empty(targets.size)
    local = np.empty(targets.size)
    linear = np.empty(targets.size)
    for i, t in enumerate(targets.tolist()):
        gi = replace(glue, tc_target=t)
        _validate_glue(gi)
        exact, lin = local_imprint(gi.eta, gi.tc_target, gi.pairing)
        dressed[i] = coherence_factor(gi) * exact
        local[i] = exact
        linear[i] = lin
    return dressed, local, linear


def fragmentation_tc(glue: SfGlueConfig = _DEFAULT_GLUE) -> float:
    """Tc* below which the LOCAL map has Tc(r) = 0 somewhere (SPECULATIVE), K.

    Tc(lambda_threshold / (1 - eta)); 0.0 when it never fragments (e.g. McMillan
    with mu* = 0).  Local limit only: the dressed maps never reach Delta = 0.
    """
    _validate_glue(glue)
    pairing = glue.pairing
    return float(tc_from_lambda(lambda_threshold(pairing) / (1.0 - glue.eta), pairing))


# ---------------------------------------------------------------------------
# Part D — hole-doped contrast (SPECULATIVE)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SfQuadratureConfig:
    """Tl2201 quadrature parameters (Ayres et al. 2021, ED Fig. 2), illustrative (SPECULATIVE)."""

    rho0: float = SF_QUAD_RHO0  # micro-Ohm cm
    a_g: float = SF_QUAD_A  # micro-Ohm cm / K
    b: float = SF_QUAD_B  # micro-Ohm cm / K^2
    alpha_kb: float = SF_QUAD_ALPHA_KB  # micro-Ohm cm / K
    gamma_mub: float = SF_QUAD_GAMMA_MUB  # micro-Ohm cm / T


_DEFAULT_QUADRATURE = SfQuadratureConfig()


def _quadrature_inputs(temperature, field, cfg: SfQuadratureConfig):
    for name in ("rho0", "a_g", "b"):
        _non_negative(name, getattr(cfg, name))
    _positive("alpha_kb", cfg.alpha_kb)
    _positive("gamma_mub", cfg.gamma_mub)
    t = _finite_array("temperature", temperature)
    if np.any(t < 0.0):
        raise ValueError("temperature must be >= 0")
    return t, _finite_array("field", field)


def quadrature_resistivity(
    temperature, field, cfg: SfQuadratureConfig = _DEFAULT_QUADRATURE
) -> float | np.ndarray:
    """rho(H, T) = rho0 + A_g T + B T^2 + sqrt((alpha kB T)^2 + (gamma muB mu0 H)^2), micro-Ohm cm.

    The hole-doped (Tl2201) quadrature form: an illustrative contrast curve (SPECULATIVE),
    not a model of LCCO.
    """
    t, b = _quadrature_inputs(temperature, field, cfg)
    return _scalar_or_array(
        cfg.rho0
        + cfg.a_g * t
        + cfg.b * t * t
        + np.sqrt((cfg.alpha_kb * t) ** 2 + (cfg.gamma_mub * b) ** 2)
    )


def quadrature_field_derivative(
    temperature, field, cfg: SfQuadratureConfig = _DEFAULT_QUADRATURE
) -> float | np.ndarray:
    """d rho / d mu0 H of the quadrature form, micro-Ohm cm / T: (gamma muB)^2 H / sqrt(...).

    An illustrative contrast (SPECULATIVE), not a model of LCCO.  A function of mu0 H / T
    alone, so it collapses against H/T by construction and saturates at gamma muB; 0 at
    T = H = 0.
    """
    t, b = _quadrature_inputs(temperature, field, cfg)
    root = np.sqrt((cfg.alpha_kb * t) ** 2 + (cfg.gamma_mub * b) ** 2)
    num = np.broadcast_to(cfg.gamma_mub**2 * b, root.shape)
    out = np.divide(num, root, out=np.zeros(root.shape), where=root > 0.0)
    return _scalar_or_array(out)
