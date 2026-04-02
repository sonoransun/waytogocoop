"""Speculative isotope-effect calculations for moire CPDM systems.

**SPECULATIVE** — These calculations use simplified models (zero-point
lattice expansion, BCS isotope effect, Debye-Waller factor) to estimate
how isotopic substitution might affect moire patterns and superconducting
gap modulation.  Results are qualitative and have not been validated
against experiment for these specific heterostructures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from waytogocoop.config import (
    AMU_TO_KG,
    DEFAULT_COHERENCE_LENGTH,
    DEFAULT_ISOTOPE_EXPONENT,
    DELTA_1,
    DELTA_2,
    DELTA_AVG,
    HBAR_J_S,
    KB_EV_K,
)
from waytogocoop.materials.isotopes import (
    ELEMENTS,
    MATERIAL_COMPOSITION,
    formula_unit_avg_mass,
)


@dataclass
class IsotopeEffects:
    """Container for all computed isotope modifications."""

    substrate_a_modified: float
    substrate_delta_a: float
    overlayer_a_modified: float
    overlayer_delta_a: float
    delta_1_modified: float
    delta_2_modified: float
    coherence_length_modified: float
    dw_factor_substrate: float
    dw_factor_overlayer: float


# ---------------------------------------------------------------------------
# Individual effect calculations
# ---------------------------------------------------------------------------


def _lattice_shift(
    formula: str,
    base_a: float,
    mass_overrides: dict[str, float] | None,
) -> tuple[float, float]:
    """Compute zero-point lattice constant shift for a material.

    Formula:
        delta_a = -a * (3 * gamma_G * k_B * T_D) / (4 * E_coh)
                  * (1 - sqrt(M_natural / M_enriched))

    The relevant mass is the average atomic mass per formula-unit atom.
    Uses element-averaged Gruneisen parameter, Debye temperature, and
    cohesive energy weighted by stoichiometry.

    Returns (modified_a, delta_a).
    """
    comp = MATERIAL_COMPOSITION[formula]
    overrides = mass_overrides or {}

    m_natural = formula_unit_avg_mass(formula, mass_overrides=None)
    m_enriched = formula_unit_avg_mass(formula, mass_overrides=overrides)

    # Stoichiometry-weighted bulk properties
    total_atoms = sum(comp.values())
    gamma_g = sum(
        count * ELEMENTS[sym].gruneisen_parameter for sym, count in comp.items()
    ) / total_atoms
    t_debye = sum(
        count * ELEMENTS[sym].debye_temperature for sym, count in comp.items()
    ) / total_atoms
    e_coh = sum(
        count * ELEMENTS[sym].cohesive_energy_ev for sym, count in comp.items()
    ) / total_atoms

    if m_enriched <= 0 or m_natural <= 0 or e_coh <= 0:
        return base_a, 0.0

    # k_B in eV/K
    prefactor = 3.0 * gamma_g * KB_EV_K * t_debye / (4.0 * e_coh)
    mass_term = 1.0 - math.sqrt(m_natural / m_enriched)

    delta_a = -base_a * prefactor * mass_term
    return base_a + delta_a, delta_a


def _gap_modification(
    delta_1: float,
    delta_2: float,
    substrate_formula: str,
    mass_overrides: dict[str, float] | None,
    alpha: float,
) -> tuple[float, float]:
    """BCS isotope effect on superconducting gap.

    Formula:
        Delta_mod = Delta_0 * (M_natural / M_enriched)^alpha

    Only applied if the substrate is a superconductor (FeTe).
    Returns (modified_delta_1, modified_delta_2).
    """
    if substrate_formula not in MATERIAL_COMPOSITION:
        return delta_1, delta_2

    m_natural = formula_unit_avg_mass(substrate_formula, mass_overrides=None)
    m_enriched = formula_unit_avg_mass(
        substrate_formula, mass_overrides=mass_overrides
    )

    if m_enriched <= 0 or m_natural <= 0:
        return delta_1, delta_2

    ratio = (m_natural / m_enriched) ** alpha
    return delta_1 * ratio, delta_2 * ratio


def _coherence_modification(
    xi_0: float,
    delta_avg_original: float,
    delta_1_mod: float,
    delta_2_mod: float,
) -> float:
    """Modified coherence length from gap change.

    xi_mod = xi_0 * (Delta_avg_original / Delta_avg_modified)
    """
    delta_avg_mod = (delta_1_mod + delta_2_mod) / 2.0
    if delta_avg_mod <= 0:
        return xi_0
    return xi_0 * (delta_avg_original / delta_avg_mod)


def _debye_waller_ratio(
    formula: str,
    lattice_type: str,
    a: float,
    mass_overrides: dict[str, float] | None,
) -> float:
    """Debye-Waller factor ratio (enriched vs natural).

    DW = exp(-G^2 * C / 2 * (1/M_natural - 1/M_enriched))
    C  = 3 * hbar^2 / (2 * k_B_J * T_D * M_amu_to_kg)

    G is the magnitude of the first reciprocal lattice vector.
    """
    comp = MATERIAL_COMPOSITION.get(formula)
    if comp is None:
        return 1.0

    overrides = mass_overrides or {}
    m_natural = formula_unit_avg_mass(formula, mass_overrides=None)
    m_enriched = formula_unit_avg_mass(formula, mass_overrides=overrides)

    if m_enriched <= 0 or m_natural <= 0:
        return 1.0

    # Average Debye temperature
    total_atoms = sum(comp.values())
    t_debye = sum(
        count * ELEMENTS[sym].debye_temperature for sym, count in comp.items()
    ) / total_atoms

    # k_B in SI (J/K)
    k_b_j = KB_EV_K * 1.602176634e-19  # convert eV/K to J/K

    # Mean-square displacement difference:
    #   delta<u^2> = 3*hbar^2 / (4 * k_B * T_D) * (1/M_nat - 1/M_enr)
    # DW_ratio = exp(-G^2 * delta<u^2>)
    c_val = 3.0 * HBAR_J_S**2 / (4.0 * k_b_j * t_debye)

    # G magnitude in SI (1/m)
    ang_to_m = 1e-10
    if lattice_type == "hexagonal":
        g_mag = 4.0 * math.pi / (a * math.sqrt(3.0))
    else:
        g_mag = 2.0 * math.pi / a
    g_si = g_mag / ang_to_m

    # Mass difference in SI (1/kg)
    inv_mass_diff = 1.0 / (m_natural * AMU_TO_KG) - 1.0 / (m_enriched * AMU_TO_KG)

    exponent = -g_si**2 * c_val * inv_mass_diff
    # Clamp to avoid overflow for extreme mass differences
    exponent = max(-100.0, min(100.0, exponent))
    return math.exp(exponent)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compute_isotope_effects(
    substrate_formula: str,
    overlayer_formula: str,
    substrate_a: float,
    overlayer_a: float,
    overlayer_lattice_type: str,
    delta_1: float = DELTA_1,
    delta_2: float = DELTA_2,
    coherence_length: float = DEFAULT_COHERENCE_LENGTH,
    mass_overrides: dict[str, float] | None = None,
    alpha: float = DEFAULT_ISOTOPE_EXPONENT,
) -> IsotopeEffects:
    """Compute all speculative isotope effects for a substrate/overlayer pair.

    Parameters
    ----------
    substrate_formula, overlayer_formula : str
        Chemical formulas (e.g. ``"FeTe"``, ``"Sb2Te3"``).
    substrate_a, overlayer_a : float
        Base lattice constants (Angstrom).
    overlayer_lattice_type : str
        ``"hexagonal"`` or ``"square"``.
    delta_1, delta_2 : float
        Superconducting gap values (meV).
    coherence_length : float
        BCS coherence length (Angstrom).
    mass_overrides : dict, optional
        Element symbol -> effective mass (amu).
    alpha : float
        BCS isotope exponent.

    Returns
    -------
    IsotopeEffects
        All computed modifications.  When *mass_overrides* is ``None`` or
        matches natural abundances, all deltas are zero / factors are 1.0.
    """
    sub_a_mod, sub_da = _lattice_shift(substrate_formula, substrate_a, mass_overrides)
    over_a_mod, over_da = _lattice_shift(overlayer_formula, overlayer_a, mass_overrides)

    d1_mod, d2_mod = _gap_modification(
        delta_1, delta_2, substrate_formula, mass_overrides, alpha
    )

    xi_mod = _coherence_modification(
        coherence_length, DELTA_AVG, d1_mod, d2_mod
    )

    dw_sub = _debye_waller_ratio(
        substrate_formula, "square", substrate_a, mass_overrides
    )
    dw_over = _debye_waller_ratio(
        overlayer_formula, overlayer_lattice_type, overlayer_a, mass_overrides
    )

    return IsotopeEffects(
        substrate_a_modified=sub_a_mod,
        substrate_delta_a=sub_da,
        overlayer_a_modified=over_a_mod,
        overlayer_delta_a=over_da,
        delta_1_modified=d1_mod,
        delta_2_modified=d2_mod,
        coherence_length_modified=xi_mod,
        dw_factor_substrate=dw_sub,
        dw_factor_overlayer=dw_over,
    )
