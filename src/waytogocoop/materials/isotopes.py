"""Isotope data for elements in the moire heterostructure materials.

Provides atomic masses, natural abundances, and bulk thermodynamic properties
(Debye temperature, Gruneisen parameter, cohesive energy) used by the
speculative isotope-effects module.

Data sources: AME2020 atomic mass evaluation, NUBASE2020 isotope tables.
Bulk properties are approximate literature values.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Isotope:
    """A single isotope of an element."""

    element: str
    mass_number: int
    atomic_mass: float  # atomic mass units (amu)
    natural_abundance: float  # fraction (0–1); 0.0 for synthetic isotopes
    half_life_s: float | None = None  # seconds; None = stable


@dataclass(frozen=True)
class ElementData:
    """Element-level data including isotopes and bulk thermodynamic properties."""

    symbol: str
    name: str
    isotopes: tuple[Isotope, ...]
    debye_temperature: float  # Kelvin
    gruneisen_parameter: float  # dimensionless
    cohesive_energy_ev: float  # eV per atom
    synthetic_isotopes: tuple[Isotope, ...] = ()  # radioactive; excluded from
    # natural averages and stable slider ranges (which iterate ``isotopes``)


@dataclass(frozen=True)
class NearestIsotope:
    """Nearest known isotope to a requested mass, with classification."""

    label: str | None  # e.g. "55Fe"; None when kind == "hypothetical"
    kind: str  # "stable" | "synthetic" | "hypothetical"
    half_life_s: float | None  # seconds; None for stable / hypothetical


# Time-unit conversions for synthetic-isotope half-lives (seconds)
_YEAR_S: float = 3.156e7
_DAY_S: float = 86400.0
_HOUR_S: float = 3600.0
_MINUTE_S: float = 60.0


# ---------------------------------------------------------------------------
# Element database
# ---------------------------------------------------------------------------

ELEMENTS: dict[str, ElementData] = {
    "Fe": ElementData(
        symbol="Fe",
        name="Iron",
        isotopes=(
            Isotope("Fe", 54, 53.9396, 0.058),
            Isotope("Fe", 56, 55.9349, 0.917),
            Isotope("Fe", 57, 56.9354, 0.022),
            Isotope("Fe", 58, 57.9333, 0.003),
        ),
        debye_temperature=260.0,
        gruneisen_parameter=1.5,
        cohesive_energy_ev=4.0,
        synthetic_isotopes=(
            Isotope("Fe", 52, 51.9481, 0.0, half_life_s=8.28 * _HOUR_S),
            Isotope("Fe", 55, 54.9383, 0.0, half_life_s=2.74 * _YEAR_S),
            Isotope("Fe", 59, 58.9349, 0.0, half_life_s=44.5 * _DAY_S),
            Isotope("Fe", 60, 59.9341, 0.0, half_life_s=2.62e6 * _YEAR_S),
        ),
    ),
    "Te": ElementData(
        symbol="Te",
        name="Tellurium",
        isotopes=(
            Isotope("Te", 122, 121.903, 0.026),
            Isotope("Te", 124, 123.903, 0.048),
            Isotope("Te", 125, 124.904, 0.071),
            Isotope("Te", 126, 125.903, 0.189),
            Isotope("Te", 128, 127.904, 0.317),
            Isotope("Te", 130, 129.906, 0.341),
        ),
        debye_temperature=165.0,
        gruneisen_parameter=1.7,
        cohesive_energy_ev=2.1,
        synthetic_isotopes=(
            Isotope("Te", 121, 120.9049, 0.0, half_life_s=19.2 * _DAY_S),
            Isotope("Te", 127, 126.9052, 0.0, half_life_s=9.35 * _HOUR_S),
            Isotope("Te", 129, 128.9066, 0.0, half_life_s=69.6 * _MINUTE_S),
            Isotope("Te", 132, 131.9085, 0.0, half_life_s=3.20 * _DAY_S),
        ),
    ),
    "Sb": ElementData(
        symbol="Sb",
        name="Antimony",
        isotopes=(
            Isotope("Sb", 121, 120.904, 0.572),
            Isotope("Sb", 123, 122.904, 0.428),
        ),
        debye_temperature=210.0,
        gruneisen_parameter=1.1,
        cohesive_energy_ev=2.7,
        synthetic_isotopes=(
            Isotope("Sb", 119, 118.9039, 0.0, half_life_s=38.2 * _HOUR_S),
            Isotope("Sb", 124, 123.9059, 0.0, half_life_s=60.2 * _DAY_S),
            Isotope("Sb", 125, 124.9053, 0.0, half_life_s=2.76 * _YEAR_S),
        ),
    ),
    "Bi": ElementData(
        symbol="Bi",
        name="Bismuth",
        isotopes=(Isotope("Bi", 209, 208.980, 1.0),),
        debye_temperature=120.0,
        gruneisen_parameter=1.2,
        cohesive_energy_ev=2.2,
        synthetic_isotopes=(
            Isotope("Bi", 207, 206.9785, 0.0, half_life_s=31.6 * _YEAR_S),
            Isotope("Bi", 208, 207.9797, 0.0, half_life_s=3.68e5 * _YEAR_S),
            Isotope("Bi", 210, 209.9841, 0.0, half_life_s=5.01 * _DAY_S),
        ),
    ),
    "C": ElementData(
        symbol="C",
        name="Carbon",
        isotopes=(
            Isotope("C", 12, 12.000, 0.9893),
            Isotope("C", 13, 13.00335, 0.0107),
        ),
        debye_temperature=2100.0,  # graphene in-plane Debye temperature
        gruneisen_parameter=1.8,
        cohesive_energy_ev=7.4,
        synthetic_isotopes=(
            Isotope("C", 11, 11.0114, 0.0, half_life_s=20.4 * _MINUTE_S),
            Isotope("C", 14, 14.0032, 0.0, half_life_s=5700.0 * _YEAR_S),
        ),
    ),
}

# ---------------------------------------------------------------------------
# Material composition (element symbol -> stoichiometric count)
# ---------------------------------------------------------------------------

MATERIAL_COMPOSITION: dict[str, dict[str, int]] = {
    "FeTe": {"Fe": 1, "Te": 1},
    "Sb2Te3": {"Sb": 2, "Te": 3},
    "Bi2Te3": {"Bi": 2, "Te": 3},
    "Sb2Te": {"Sb": 2, "Te": 1},
    "Graphene": {"C": 2},  # 2 C atoms per hexagonal unit cell (A and B sublattices)
    "Graphene-AB": {"C": 4},   # 2 C atoms per layer per cell, 2 layers
    "Graphene-ABA": {"C": 6},  # 2 C atoms per layer per cell, 3 layers
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def get_element(symbol: str) -> ElementData:
    """Look up element data by symbol.

    Raises ``KeyError`` if the element is not in the database.
    """
    if symbol not in ELEMENTS:
        raise KeyError(f"Unknown element '{symbol}'. Known: {list(ELEMENTS.keys())}")
    return ELEMENTS[symbol]


def get_composition(formula: str) -> dict[str, int]:
    """Return the stoichiometric composition for a material formula.

    Raises ``KeyError`` if the formula is not recognised.
    """
    return MATERIAL_COMPOSITION[formula]


def natural_average_mass(symbol: str) -> float:
    """Abundance-weighted average atomic mass for an element (amu)."""
    elem = ELEMENTS[symbol]
    return sum(iso.atomic_mass * iso.natural_abundance for iso in elem.isotopes)


def te_125_spin_fraction(te_mass_override: float | None = None) -> float:
    """Estimate the ¹²⁵Te nuclear spin fraction for a given Te mass setting.

    ¹²⁵Te (I=1/2) is the only spin-bearing stable Te isotope. A target mass
    between two adjacent STABLE isotopes (m_lo, m_hi) is treated as a binary
    two-isotope mixture with

        weight_lo = (m_hi - target) / (m_hi - m_lo)

    and the ¹²⁵Te fraction is the mixture weight belonging to the 124.904 amu
    endpoint if one of the brackets is ¹²⁵Te, else 0.0.  Below the lightest /
    above the heaviest stable isotope, the composition is taken as pure
    lightest / heaviest isotope.

    Note: an override equal to the natural average mass (~126.62 amu) is
    interpreted as an enriched two-isotope blend of ¹²⁶Te/¹²⁸Te, NOT the
    natural composition — pass ``None`` for natural abundance (0.071).
    """
    te = ELEMENTS["Te"]
    if te_mass_override is None:
        return 0.071  # natural abundance of ¹²⁵Te

    target = te_mass_override
    te_125_mass = 124.904

    # Stable isotope masses only (synthetics live in synthetic_isotopes)
    masses = sorted(iso.atomic_mass for iso in te.isotopes)
    if target <= masses[0]:
        return 1.0 if abs(masses[0] - te_125_mass) < 0.5 else 0.0
    if target >= masses[-1]:
        return 1.0 if abs(masses[-1] - te_125_mass) < 0.5 else 0.0

    # Two adjacent stable isotopes bracketing the target: binary mixture
    for m_lo, m_hi in zip(masses, masses[1:], strict=False):
        if m_lo <= target <= m_hi:
            weight_lo = (m_hi - target) / (m_hi - m_lo)
            if abs(m_lo - te_125_mass) < 1e-6:
                return weight_lo
            if abs(m_hi - te_125_mass) < 1e-6:
                return 1.0 - weight_lo
            return 0.0
    return 0.0  # unreachable given the range checks above


def nearest_isotope_info(symbol: str, mass_amu: float) -> NearestIsotope:
    """Classify a mass as the nearest stable/synthetic isotope, or hypothetical.

    Searches both the stable and synthetic isotope lists of *symbol*.  If the
    nearest known isotope lies within 0.25 amu of *mass_amu*, returns its label
    (e.g. ``"55Fe"``), its kind (``"stable"`` or ``"synthetic"``), and its
    half-life (``None`` for stable).  Otherwise the mass is classified as
    ``"hypothetical"`` with no label.
    """
    elem = get_element(symbol)
    candidates = elem.isotopes + elem.synthetic_isotopes
    nearest = min(candidates, key=lambda iso: abs(iso.atomic_mass - mass_amu))
    if abs(nearest.atomic_mass - mass_amu) > 0.25:
        return NearestIsotope(label=None, kind="hypothetical", half_life_s=None)
    kind = "stable" if nearest.half_life_s is None else "synthetic"
    return NearestIsotope(
        label=f"{nearest.mass_number}{symbol}",
        kind=kind,
        half_life_s=nearest.half_life_s,
    )


def humanize_half_life(seconds: float) -> str:
    """Human-readable half-life string, e.g. "69.6 min", "8.3 h", "2.7 y", "2.6 My"."""
    if seconds < 2.0 * _HOUR_S:
        return f"{seconds / _MINUTE_S:.1f} min"
    if seconds < 2.0 * _DAY_S:
        return f"{seconds / _HOUR_S:.1f} h"
    if seconds < 2.0 * _YEAR_S:
        return f"{seconds / _DAY_S:.1f} d"
    years = seconds / _YEAR_S
    if years < 1.0e3:
        return f"{years:.1f} y"
    if years < 1.0e6:
        return f"{years / 1.0e3:.1f} ky"
    return f"{years / 1.0e6:.1f} My"


def formula_unit_avg_mass(
    formula: str,
    mass_overrides: dict[str, float] | None = None,
) -> float:
    """Average atomic mass per atom in a formula unit.

    Parameters
    ----------
    formula : str
        Material formula, e.g. ``"FeTe"`` or ``"Sb2Te3"``.
    mass_overrides : dict, optional
        Element symbol -> effective mass (amu).  Falls back to the
        natural-abundance average for elements not in the dict.

    Returns
    -------
    float
        Weighted average mass per atom (amu).
    """
    if formula not in MATERIAL_COMPOSITION:
        raise ValueError(
            f"Unknown material formula '{formula}'. "
            f"Known: {list(MATERIAL_COMPOSITION.keys())}"
        )
    comp = MATERIAL_COMPOSITION[formula]
    overrides = mass_overrides or {}
    total_mass = 0.0
    total_atoms = 0
    for elem, count in comp.items():
        m = overrides.get(elem, natural_average_mass(elem))
        total_mass += count * m
        total_atoms += count
    return total_mass / total_atoms
