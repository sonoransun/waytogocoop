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
    natural_abundance: float  # fraction (0–1)


@dataclass(frozen=True)
class ElementData:
    """Element-level data including isotopes and bulk thermodynamic properties."""

    symbol: str
    name: str
    isotopes: tuple[Isotope, ...]
    debye_temperature: float  # Kelvin
    gruneisen_parameter: float  # dimensionless
    cohesive_energy_ev: float  # eV per atom


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
    ),
    "Bi": ElementData(
        symbol="Bi",
        name="Bismuth",
        isotopes=(Isotope("Bi", 209, 208.980, 1.0),),
        debye_temperature=120.0,
        gruneisen_parameter=1.2,
        cohesive_energy_ev=2.2,
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
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def get_element(symbol: str) -> ElementData:
    """Look up element data by symbol.

    Raises ``KeyError`` if the element is not in the database.
    """
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
    comp = MATERIAL_COMPOSITION[formula]
    overrides = mass_overrides or {}
    total_mass = 0.0
    total_atoms = 0
    for elem, count in comp.items():
        m = overrides.get(elem, natural_average_mass(elem))
        total_mass += count * m
        total_atoms += count
    return total_mass / total_atoms
