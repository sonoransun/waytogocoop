"""Speculative isotope-effect calculations for moire CPDM systems.

**SPECULATIVE** — These calculations use simplified models (zero-point
lattice expansion, BCS isotope effect, Debye-Waller factor) to estimate
how isotopic substitution might affect moire patterns and superconducting
gap modulation.  Results are qualitative and have not been validated
against experiment for these specific heterostructures.

Literature basis for the BCS isotope exponent (alpha):
  - FeSe: alpha_Fe = 0.81 ± 0.15 (Khasanov et al., arXiv:1002.2510)
  - SmFeAsO_{1-x}F_x: alpha_Fe ~ 0.35 (Liu et al., Nature 459, 64, 2009)
  - (Ba,K)Fe2As2: alpha_Fe = -0.18 — inverse effect (Shirage et al., PRL 103, 257003)
  - Corrected consensus: alpha ~ 0.35-0.4 (PRB 82, 212505)
  - No Te isotope data exists for FeTe; this is a key experimental gap.
The ground state is theoretically a "phonon-dressed unconventional superconductor"
where electronic pairing dominates but phonons dress the interaction (PMC6447578).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

from waytogocoop.config import (
    AMU_TO_KG,
    ANGSTROM_TO_M,
    DEFAULT_COHERENCE_LENGTH,
    DEFAULT_ISOTOPE_EXPONENT,
    DELTA_1,
    DELTA_2,
    DELTA_AVG,
    EXPONENT_CLAMP,
    HBAR_J_S,
    KB_EV_K,
)
from waytogocoop.materials.isotopes import (
    ELEMENTS,
    MATERIAL_COMPOSITION,
    formula_unit_avg_mass,
    te_125_spin_fraction,
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
    theta_d_substrate: float  # isotope-shifted Debye temperature (K)
    theta_d_overlayer: float  # isotope-shifted Debye temperature (K)
    te_125_spin_fraction: float  # fraction of Te that is ¹²⁵Te (I=1/2)


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

    Heavier enrichment contracts the lattice via reduced zero-point
    anharmonic expansion.  The prefactor uses the natural T_D (leading order).

    The relevant mass is the average atomic mass per formula-unit atom.
    Uses element-averaged Gruneisen parameter, Debye temperature, and
    cohesive energy weighted by stoichiometry.

    Returns (modified_a, delta_a).
    """
    if base_a <= 0:
        raise ValueError("base_a must be positive")
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
        warnings.warn(
            f"Formula '{substrate_formula}' not in MATERIAL_COMPOSITION; "
            "returning unmodified gaps",
            stacklevel=2,
        )
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
    """Debye-Waller factor ratio (enriched vs natural), zero-point only.

    Per-component zero-point DW exponent for an atom of mass M:

        2W(M) = G^2 * C / sqrt(M * M_nat),   C = 3*hbar^2 / (4 * k_B * Theta_nat)

    which follows from the zero-point mean-square displacement
    <u^2>_zp = 9*hbar^2 / (4 * M * k_B * Theta_D(M)), the isotropic
    projection <u_G^2> = <u^2>/3, and the Debye-temperature mass
    co-variation Theta_D(M) = Theta_nat * sqrt(M_nat / M).  Hence

        DW_ratio = exp(-G^2 * C * (1/sqrt(M_enr * M_nat) - 1/M_nat))

    The model is zero-point only — and that is the right model here: the
    classical (high-T) thermal MSD 3*k_B*T/(M*omega_bar^2) is
    mass-independent because M*omega_bar^2 does not depend on isotope mass,
    so the isotope contrast in the DW factor is purely a quantum zero-point
    effect.  Heavier enrichment gives ratio > 1 (less zero-point smearing,
    sharper potential); lighter gives ratio < 1.

    G is the magnitude of the first reciprocal lattice vector; all
    quantities in SI (masses in kg via AMU_TO_KG).
    """
    comp = MATERIAL_COMPOSITION.get(formula)
    if comp is None:
        return 1.0

    overrides = mass_overrides or {}
    m_natural = formula_unit_avg_mass(formula, mass_overrides=None)
    m_enriched = formula_unit_avg_mass(formula, mass_overrides=overrides)

    if m_enriched <= 0 or m_natural <= 0:
        return 1.0

    # Average natural Debye temperature (Theta_nat)
    total_atoms = sum(comp.values())
    t_debye = sum(
        count * ELEMENTS[sym].debye_temperature for sym, count in comp.items()
    ) / total_atoms

    # k_B in SI (J/K)
    k_b_j = KB_EV_K * 1.602176634e-19  # convert eV/K to J/K

    # C = 3*hbar^2 / (4 * k_B * Theta_nat)
    c_val = 3.0 * HBAR_J_S**2 / (4.0 * k_b_j * t_debye)

    # G magnitude in SI (1/m)
    if lattice_type == "hexagonal":
        g_mag = 4.0 * math.pi / (a * math.sqrt(3.0))
    else:
        g_mag = 2.0 * math.pi / a
    g_si = g_mag / ANGSTROM_TO_M

    # Zero-point exponent difference in SI (1/kg)
    m_nat_kg = m_natural * AMU_TO_KG
    m_enr_kg = m_enriched * AMU_TO_KG
    mass_term = 1.0 / math.sqrt(m_enr_kg * m_nat_kg) - 1.0 / m_nat_kg

    exponent = -g_si**2 * c_val * mass_term
    # Clamp to prevent math.exp overflow for extreme mass differences
    exponent = max(-EXPONENT_CLAMP, min(EXPONENT_CLAMP, exponent))
    return math.exp(exponent)


def _isotope_shifted_debye_temperature(
    formula: str,
    mass_overrides: dict[str, float] | None,
) -> float:
    """Isotope-shifted Debye temperature (Kelvin).

    Theta_enr = Theta_nat * sqrt(M_nat / M_enr), with Theta_nat the
    stoichiometry-weighted average Debye temperature and the masses the
    formula-unit per-atom averages.
    """
    comp = MATERIAL_COMPOSITION[formula]
    total_atoms = sum(comp.values())
    theta_nat = sum(
        count * ELEMENTS[sym].debye_temperature for sym, count in comp.items()
    ) / total_atoms

    m_natural = formula_unit_avg_mass(formula, mass_overrides=None)
    m_enriched = formula_unit_avg_mass(formula, mass_overrides=mass_overrides)
    if m_enriched <= 0 or m_natural <= 0:
        return theta_nat
    return theta_nat * math.sqrt(m_natural / m_enriched)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compute_isotope_effects(
    substrate_formula: str,
    overlayer_formula: str,
    substrate_a: float,
    overlayer_a: float,
    overlayer_lattice_type: str,
    substrate_lattice_type: str = "square",
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
    if substrate_a <= 0:
        raise ValueError("substrate_a must be positive")
    if overlayer_a <= 0:
        raise ValueError("overlayer_a must be positive")
    if delta_1 < 0:
        raise ValueError("delta_1 must be >= 0")
    if delta_2 < 0:
        raise ValueError("delta_2 must be >= 0")
    if coherence_length <= 0:
        raise ValueError("coherence_length must be positive")
    if overlayer_lattice_type not in ("hexagonal", "square"):
        raise ValueError(
            f"overlayer_lattice_type must be 'hexagonal' or 'square', "
            f"got '{overlayer_lattice_type}'"
        )
    if substrate_lattice_type not in ("hexagonal", "square"):
        raise ValueError(
            f"substrate_lattice_type must be 'hexagonal' or 'square', "
            f"got '{substrate_lattice_type}'"
        )

    sub_a_mod, sub_da = _lattice_shift(substrate_formula, substrate_a, mass_overrides)
    over_a_mod, over_da = _lattice_shift(overlayer_formula, overlayer_a, mass_overrides)

    d1_mod, d2_mod = _gap_modification(
        delta_1, delta_2, substrate_formula, mass_overrides, alpha
    )

    xi_mod = _coherence_modification(
        coherence_length, DELTA_AVG, d1_mod, d2_mod
    )

    dw_sub = _debye_waller_ratio(
        substrate_formula, substrate_lattice_type, substrate_a, mass_overrides
    )
    dw_over = _debye_waller_ratio(
        overlayer_formula, overlayer_lattice_type, overlayer_a, mass_overrides
    )

    theta_sub = _isotope_shifted_debye_temperature(substrate_formula, mass_overrides)
    theta_over = _isotope_shifted_debye_temperature(overlayer_formula, mass_overrides)

    overrides = mass_overrides or {}
    spin_frac = te_125_spin_fraction(overrides.get("Te"))

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
        theta_d_substrate=theta_sub,
        theta_d_overlayer=theta_over,
        te_125_spin_fraction=spin_frac,
    )
