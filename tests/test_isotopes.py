"""Tests for isotope data and isotope-effect computations."""

from __future__ import annotations

import pytest

from waytogocoop.materials.isotopes import (
    ELEMENTS,
    formula_unit_avg_mass,
    get_composition,
    get_element,
    natural_average_mass,
)
from waytogocoop.computation.isotope_effects import compute_isotope_effects
from waytogocoop.config import DELTA_1, DELTA_2


# ---------------------------------------------------------------------------
# Isotope data integrity
# ---------------------------------------------------------------------------


class TestIsotopeData:
    @pytest.mark.parametrize("symbol", ["Fe", "Te", "Sb", "Bi"])
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

    def test_get_element(self):
        assert get_element("Fe").symbol == "Fe"
        with pytest.raises(KeyError):
            get_element("Zz")

    def test_get_composition(self):
        assert get_composition("FeTe") == {"Fe": 1, "Te": 1}
        assert get_composition("Sb2Te3") == {"Sb": 2, "Te": 3}

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
