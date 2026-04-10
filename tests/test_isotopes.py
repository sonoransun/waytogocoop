"""Tests for isotope data and isotope-effect computations."""

from __future__ import annotations

import numpy as np
import pytest

from waytogocoop.computation.isotope_effects import compute_isotope_effects
from waytogocoop.config import DELTA_1, DELTA_2
from waytogocoop.materials.isotopes import (
    ELEMENTS,
    formula_unit_avg_mass,
    get_composition,
    get_element,
    natural_average_mass,
)

# ---------------------------------------------------------------------------
# Isotope data integrity
# ---------------------------------------------------------------------------


class TestIsotopeData:
    @pytest.mark.parametrize("symbol", ["Fe", "Te", "Sb", "Bi", "C"])
    def test_abundances_sum_to_one(self, symbol):
        elem = ELEMENTS[symbol]
        total = sum(iso.natural_abundance for iso in elem.isotopes)
        assert total == pytest.approx(1.0, abs=0.01)

    def test_natural_average_mass_fe(self):
        # IUPAC standard atomic weight: 55.845
        assert natural_average_mass("Fe") == pytest.approx(55.845, abs=0.1)

    def test_natural_average_mass_te(self):
        # IUPAC: 127.60 — our database omits minor isotopes (120Te, 123Te)
        assert natural_average_mass("Te") == pytest.approx(127.60, abs=1.1)

    def test_natural_average_mass_sb(self):
        # IUPAC: 121.760
        assert natural_average_mass("Sb") == pytest.approx(121.76, abs=0.1)

    def test_natural_average_mass_bi(self):
        # Monoisotopic: 208.980
        assert natural_average_mass("Bi") == pytest.approx(208.98, abs=0.01)

    def test_natural_average_mass_c(self):
        # IUPAC standard atomic weight: 12.011
        assert natural_average_mass("C") == pytest.approx(12.011, abs=0.01)

    def test_get_element(self):
        assert get_element("Fe").symbol == "Fe"
        assert get_element("C").symbol == "C"
        with pytest.raises(KeyError):
            get_element("Zz")

    def test_get_composition(self):
        assert get_composition("FeTe") == {"Fe": 1, "Te": 1}
        assert get_composition("Sb2Te3") == {"Sb": 2, "Te": 3}
        # 2 carbon atoms per graphene hexagonal unit cell (A/B sublattices).
        assert get_composition("Graphene") == {"C": 2}

    def test_formula_unit_avg_mass_fete(self):
        m = formula_unit_avg_mass("FeTe")
        expected = (natural_average_mass("Fe") + natural_average_mass("Te")) / 2.0
        assert m == pytest.approx(expected, rel=1e-10)

    def test_formula_unit_avg_mass_with_override(self):
        m = formula_unit_avg_mass("FeTe", mass_overrides={"Fe": 54.0})
        expected = (54.0 + natural_average_mass("Te")) / 2.0
        assert m == pytest.approx(expected, rel=1e-10)


# ---------------------------------------------------------------------------
# Isotope effects — identity (natural masses)
# ---------------------------------------------------------------------------


@pytest.mark.speculative
class TestIsotopeEffectsIdentity:
    """With no mass overrides, all modifications should be zero/identity."""

    def setup_method(self):
        self.effects = compute_isotope_effects(
            substrate_formula="FeTe",
            overlayer_formula="Sb2Te3",
            substrate_a=3.82,
            overlayer_a=4.264,
            overlayer_lattice_type="hexagonal",
        )

    def test_zero_substrate_shift(self):
        assert self.effects.substrate_delta_a == pytest.approx(0.0, abs=1e-12)

    def test_zero_overlayer_shift(self):
        assert self.effects.overlayer_delta_a == pytest.approx(0.0, abs=1e-12)

    def test_identity_gaps(self):
        assert self.effects.delta_1_modified == pytest.approx(DELTA_1, rel=1e-10)
        assert self.effects.delta_2_modified == pytest.approx(DELTA_2, rel=1e-10)

    def test_identity_coherence(self):
        assert self.effects.coherence_length_modified == pytest.approx(20.0, rel=1e-10)

    def test_identity_dw(self):
        assert self.effects.dw_factor_substrate == pytest.approx(1.0, abs=1e-10)
        assert self.effects.dw_factor_overlayer == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Isotope effects — physics sign & magnitude
# ---------------------------------------------------------------------------


@pytest.mark.speculative
class TestIsotopeEffectsPhysics:
    def test_heavier_isotope_contracts_lattice(self):
        effects = compute_isotope_effects(
            substrate_formula="FeTe",
            overlayer_formula="Sb2Te3",
            substrate_a=3.82,
            overlayer_a=4.264,
            overlayer_lattice_type="hexagonal",
            mass_overrides={"Fe": 57.9333},  # Fe-58 (heavier)
        )
        assert effects.substrate_delta_a < 0.0

    def test_lighter_isotope_expands_lattice(self):
        effects = compute_isotope_effects(
            substrate_formula="FeTe",
            overlayer_formula="Sb2Te3",
            substrate_a=3.82,
            overlayer_a=4.264,
            overlayer_lattice_type="hexagonal",
            mass_overrides={"Fe": 53.9396},  # Fe-54 (lighter)
        )
        assert effects.substrate_delta_a > 0.0

    def test_lighter_isotope_increases_gap(self):
        effects = compute_isotope_effects(
            substrate_formula="FeTe",
            overlayer_formula="Sb2Te3",
            substrate_a=3.82,
            overlayer_a=4.264,
            overlayer_lattice_type="hexagonal",
            mass_overrides={"Fe": 53.9396},
        )
        assert effects.delta_1_modified > DELTA_1
        assert effects.delta_2_modified > DELTA_2

    def test_larger_gap_shortens_coherence(self):
        effects = compute_isotope_effects(
            substrate_formula="FeTe",
            overlayer_formula="Sb2Te3",
            substrate_a=3.82,
            overlayer_a=4.264,
            overlayer_lattice_type="hexagonal",
            mass_overrides={"Fe": 53.9396},
        )
        assert effects.coherence_length_modified < 20.0

    def test_lattice_shift_magnitude(self):
        """Shift should be order 1e-5 to 1e-2 Angstrom."""
        effects = compute_isotope_effects(
            substrate_formula="FeTe",
            overlayer_formula="Sb2Te3",
            substrate_a=3.82,
            overlayer_a=4.264,
            overlayer_lattice_type="hexagonal",
            mass_overrides={"Fe": 53.9396},
        )
        abs_da = abs(effects.substrate_delta_a)
        assert 1e-6 < abs_da < 0.1

    def test_dw_factor_near_unity(self):
        """DW factor should be close to 1 for small mass changes."""
        effects = compute_isotope_effects(
            substrate_formula="FeTe",
            overlayer_formula="Sb2Te3",
            substrate_a=3.82,
            overlayer_a=4.264,
            overlayer_lattice_type="hexagonal",
            mass_overrides={"Fe": 53.9396},
        )
        assert 0.9 < effects.dw_factor_substrate < 1.1
        assert 0.9 < effects.dw_factor_overlayer < 1.1

    def test_alpha_zero_gives_no_gap_change(self):
        """alpha=0 means no isotope effect on gap."""
        effects = compute_isotope_effects(
            substrate_formula="FeTe",
            overlayer_formula="Sb2Te3",
            substrate_a=3.82,
            overlayer_a=4.264,
            overlayer_lattice_type="hexagonal",
            mass_overrides={"Fe": 53.9396},
            alpha=0.0,
        )
        assert effects.delta_1_modified == pytest.approx(DELTA_1, rel=1e-10)

    def test_opposite_shifts_for_light_vs_heavy(self):
        """Fe-54 (lighter) expands lattice; Fe-58 (heavier) contracts it."""
        eff_54 = compute_isotope_effects(
            substrate_formula="FeTe",
            overlayer_formula="Sb2Te3",
            substrate_a=3.82,
            overlayer_a=4.264,
            overlayer_lattice_type="hexagonal",
            mass_overrides={"Fe": 53.9396},
        )
        eff_58 = compute_isotope_effects(
            substrate_formula="FeTe",
            overlayer_formula="Sb2Te3",
            substrate_a=3.82,
            overlayer_a=4.264,
            overlayer_lattice_type="hexagonal",
            mass_overrides={"Fe": 57.9333},
        )
        assert eff_54.substrate_delta_a > 0  # lighter -> expanded
        assert eff_58.substrate_delta_a < 0  # heavier -> contracted
        # Both should produce measurable shifts
        assert abs(eff_54.substrate_delta_a) > 1e-6
        assert abs(eff_58.substrate_delta_a) > 1e-6


class TestIsotopeValidation:
    """Edge case and input validation tests for isotope effects."""

    def test_compute_isotope_effects_negative_substrate_a_raises(self):
        with pytest.raises(ValueError):
            compute_isotope_effects(
                substrate_formula="FeTe",
                overlayer_formula="Sb2Te3",
                substrate_a=-1.0,
                overlayer_a=4.264,
                overlayer_lattice_type="hexagonal",
            )

    def test_compute_isotope_effects_negative_delta_raises(self):
        with pytest.raises(ValueError):
            compute_isotope_effects(
                substrate_formula="FeTe",
                overlayer_formula="Sb2Te3",
                substrate_a=3.82,
                overlayer_a=4.264,
                overlayer_lattice_type="hexagonal",
                delta_1=-1.0,
            )

    def test_compute_isotope_effects_zero_coherence_raises(self):
        with pytest.raises(ValueError):
            compute_isotope_effects(
                substrate_formula="FeTe",
                overlayer_formula="Sb2Te3",
                substrate_a=3.82,
                overlayer_a=4.264,
                overlayer_lattice_type="hexagonal",
                coherence_length=0.0,
            )

    def test_formula_unit_avg_mass_unknown_formula_raises(self):
        with pytest.raises(ValueError):
            formula_unit_avg_mass("XYZ")

    def test_extreme_mass_ratio_light(self):
        """Very light Fe mass override should not crash."""
        effects = compute_isotope_effects(
            substrate_formula="FeTe",
            overlayer_formula="Sb2Te3",
            substrate_a=3.82,
            overlayer_a=4.264,
            overlayer_lattice_type="hexagonal",
            mass_overrides={"Fe": 1.0},
        )
        assert np.isfinite(effects.substrate_delta_a)
        assert np.isfinite(effects.dw_factor_substrate)

    def test_extreme_mass_ratio_heavy(self):
        """Very heavy Fe mass override should not crash."""
        effects = compute_isotope_effects(
            substrate_formula="FeTe",
            overlayer_formula="Sb2Te3",
            substrate_a=3.82,
            overlayer_a=4.264,
            overlayer_lattice_type="hexagonal",
            mass_overrides={"Fe": 10000.0},
        )
        assert np.isfinite(effects.substrate_delta_a)
        assert np.isfinite(effects.dw_factor_substrate)

    def test_dw_exponent_clamping(self):
        """Extreme mass difference should still produce finite DW factor."""
        effects = compute_isotope_effects(
            substrate_formula="FeTe",
            overlayer_formula="Sb2Te3",
            substrate_a=3.82,
            overlayer_a=4.264,
            overlayer_lattice_type="hexagonal",
            mass_overrides={"Fe": 0.1},
        )
        assert np.isfinite(effects.dw_factor_substrate)
        assert effects.dw_factor_substrate > 0
