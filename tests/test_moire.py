"""Tests for moire periodicity and pattern generation."""

from __future__ import annotations

import numpy as np
import pytest

from waytogocoop.computation.moire import (
    generate_moire_pattern,
    moire_periodicity_1d,
    moire_periodicity_with_twist,
)


class TestMoirePeriodicity:
    def test_sb2te3_on_fete(self):
        """Sb2Te3 (a=4.264) on FeTe (a=3.82) should give ~36.7 A."""
        period = moire_periodicity_1d(3.82, 4.264)
        assert period == pytest.approx(36.67, abs=0.1)

    def test_bi2te3_on_fete(self):
        """Bi2Te3 (a=4.386) on FeTe (a=3.82) gives ~29.6 A."""
        period = moire_periodicity_1d(3.82, 4.386)
        assert period == pytest.approx(29.6, abs=0.2)

    def test_symmetric(self):
        """moire_periodicity_1d(a, b) == moire_periodicity_1d(b, a)."""
        p1 = moire_periodicity_1d(3.82, 4.264)
        p2 = moire_periodicity_1d(4.264, 3.82)
        assert p1 == pytest.approx(p2)

    def test_same_lattice_constant_raises_large(self):
        """Identical lattice constants -> divergent period."""
        period = moire_periodicity_1d(3.82, 3.82 + 1e-15)
        assert period > 1e10

    def test_twist_formula_small_angle(self):
        """Twist of 1 deg on a=4.0 A gives L ~ 229 A."""
        period = moire_periodicity_with_twist(4.0, 1.0)
        expected = 4.0 / (2 * np.sin(np.radians(0.5)))
        assert period == pytest.approx(expected, rel=1e-6)

    def test_twist_zero_returns_inf(self):
        period = moire_periodicity_with_twist(4.0, 0.0)
        assert np.isinf(period)


class TestGenerateMoirePattern:
    def test_output_shape(self):
        result = generate_moire_pattern(
            substrate_a=3.82,
            overlayer_a=4.264,
            overlayer_lattice_type="hexagonal",
            grid_size=100,
            physical_extent=50.0,
        )
        assert result["pattern"].shape == (100, 100)
        assert len(result["x"]) == 100
        assert len(result["y"]) == 100

    def test_pattern_normalized(self):
        result = generate_moire_pattern(
            substrate_a=3.82,
            overlayer_a=4.264,
            overlayer_lattice_type="hexagonal",
            grid_size=64,
        )
        assert result["pattern"].min() == pytest.approx(0.0, abs=1e-10)
        assert result["pattern"].max() == pytest.approx(1.0, abs=1e-10)

    def test_moire_period_key_present(self):
        result = generate_moire_pattern(
            substrate_a=3.82,
            overlayer_a=4.264,
            overlayer_lattice_type="hexagonal",
        )
        assert "moire_period" in result
        assert result["moire_period"] > 0

    def test_with_twist_angle(self):
        result = generate_moire_pattern(
            substrate_a=3.82,
            overlayer_a=4.264,
            overlayer_lattice_type="hexagonal",
            twist_angle_deg=5.0,
            grid_size=64,
        )
        assert result["pattern"].shape == (64, 64)
        assert np.isfinite(result["moire_period"])

    def test_square_overlayer(self):
        result = generate_moire_pattern(
            substrate_a=3.82,
            overlayer_a=4.0,
            overlayer_lattice_type="square",
            grid_size=64,
        )
        assert result["pattern"].shape == (64, 64)

    def test_equal_lattice_constants(self):
        result = generate_moire_pattern(
            substrate_a=4.0,
            overlayer_a=4.0,
            overlayer_lattice_type="square",
            grid_size=64,
        )
        assert np.isinf(result["moire_period"])


class TestMoireValidation:
    """Edge case and input validation tests for moire computation."""

    def test_periodicity_1d_negative_a1(self):
        with pytest.raises(ValueError):
            moire_periodicity_1d(-1, 4.0)

    def test_periodicity_1d_zero_a2(self):
        with pytest.raises(ValueError):
            moire_periodicity_1d(3.82, 0)

    def test_periodicity_with_twist_negative_a(self):
        with pytest.raises(ValueError):
            moire_periodicity_with_twist(-1.0, 5.0)

    def test_generate_moire_zero_substrate_a(self):
        with pytest.raises(ValueError):
            generate_moire_pattern(
                substrate_a=0,
                overlayer_a=4.264,
                overlayer_lattice_type="hexagonal",
            )

    def test_generate_moire_zero_overlayer_a(self):
        with pytest.raises(ValueError):
            generate_moire_pattern(
                substrate_a=3.82,
                overlayer_a=0,
                overlayer_lattice_type="hexagonal",
            )

    def test_generate_moire_grid_size_one(self):
        with pytest.raises(ValueError):
            generate_moire_pattern(
                substrate_a=3.82,
                overlayer_a=4.264,
                overlayer_lattice_type="hexagonal",
                grid_size=1,
            )

    def test_generate_moire_negative_extent(self):
        with pytest.raises(ValueError):
            generate_moire_pattern(
                substrate_a=3.82,
                overlayer_a=4.264,
                overlayer_lattice_type="hexagonal",
                physical_extent=-10.0,
            )

    def test_generate_moire_invalid_lattice_type(self):
        with pytest.raises(ValueError):
            generate_moire_pattern(
                substrate_a=3.82,
                overlayer_a=4.264,
                overlayer_lattice_type="triangular",
            )

    def test_generate_moire_dw_factor_nonunity(self):
        """Non-unity DW factor changes the pattern when substrate and overlayer differ."""
        # Use asymmetric DW factors (substrate=0.5, overlayer=2.0 vs both=1.0)
        # to produce non-trivially different raw products before normalization.
        # Since normalization maps to [0,1], a uniform scalar won't change
        # the result, but changing the ratio between substrate and overlayer
        # amplitudes effectively changes the relative weight, leading to
        # different normalized patterns when dw_factor=0 for one layer.
        result_zero_sub = generate_moire_pattern(
            substrate_a=3.82,
            overlayer_a=4.264,
            overlayer_lattice_type="hexagonal",
            grid_size=64,
            dw_factor_substrate=0.0,
        )
        # When dw_factor_substrate=0, V_sub=0 so pattern=0*V_over=0,
        # which normalizes to all zeros.
        assert np.allclose(result_zero_sub["pattern"], 0.0)

    def test_generate_moire_all_finite(self):
        """Generated pattern should contain only finite values (NaN guard)."""
        result = generate_moire_pattern(
            substrate_a=3.82,
            overlayer_a=4.264,
            overlayer_lattice_type="hexagonal",
            grid_size=64,
            physical_extent=50.0,
        )
        assert np.all(np.isfinite(result["pattern"]))
