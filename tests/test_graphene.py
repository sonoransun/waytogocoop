"""Tests for twisted/stacked graphene patterns and flat-band superconductivity."""

from __future__ import annotations

import numpy as np
import pytest

from waytogocoop.computation.curvature import (
    CurvatureConfig,
    displacement_field,
    height_field,
)
from waytogocoop.computation.graphene import (
    compute_flat_band_sc,
    dirac_velocity_ratio,
    filling_dome_factor,
    generate_stack_pattern,
    generate_stack_pattern_v2,
    generate_supermoire_pattern,
    layer_potential,
    layer_potential_honeycomb,
    magic_angle_deg,
    moire_period_from_g_shells,
    stack_layers,
    strained_g_vectors,
)
from waytogocoop.computation.moire import _reciprocal_g_vectors_hexagonal
from waytogocoop.computation.superconducting import cpdm_amplitude, gap_modulation
from waytogocoop.config import DELTA_TBG_MAX, XI_TBG
from waytogocoop.materials.lattice import apply_rotation


class TestStackLayers:
    @pytest.mark.parametrize(
        ("stacking", "expected"),
        [
            ("AA", 2),
            ("AB", 2),
            ("twisted_bilayer", 2),
            ("ABA", 3),
            ("ABC", 3),
            ("alternating_trilayer", 3),
        ],
    )
    def test_layer_counts(self, stacking, expected):
        assert len(stack_layers(stacking)) == expected

    def test_stacking_indices(self):
        assert [layer.stacking_index for layer in stack_layers("AB")] == [0, 1]
        assert [layer.stacking_index for layer in stack_layers("ABA")] == [0, 1, 0]
        assert [layer.stacking_index for layer in stack_layers("ABC")] == [0, 1, 2]

    def test_twist_assignment(self):
        twists = [layer.twist_deg for layer in stack_layers("twisted_bilayer", 1.1)]
        assert twists == [0.0, 1.1]
        twists = [layer.twist_deg for layer in stack_layers("alternating_trilayer", 1.5)]
        assert twists == [0.0, 1.5, 0.0]

    def test_unknown_stacking_raises(self):
        with pytest.raises(ValueError, match="Unknown stacking"):
            stack_layers("AAB")


class TestLayerPotential:
    def test_origin_a_registry(self):
        """All 6 cosines are 1 at the origin for the A registry."""
        v = layer_potential(np.array([0.0]), np.array([0.0]), stacking_index=0)
        assert v[0, 0] == pytest.approx(6.0, abs=1e-12)

    def test_origin_b_registry(self):
        """B registry at the origin: 4*cos(2*pi/3) + 2*cos(0) = 0."""
        v = layer_potential(np.array([0.0]), np.array([0.0]), stacking_index=1)
        assert v[0, 0] == pytest.approx(0.0, abs=1e-12)

    def test_lattice_periodicity(self):
        """V is invariant under the real-space primitive a*(sqrt(3)/2, -1/2).

        Pins the dual-frame convention of the angle-ordered G shell.
        """
        a = 2.46
        shift = a * np.array([np.sqrt(3.0) / 2.0, -0.5])
        points = [(0.7, -1.3), (3.1, 2.2), (-5.4, 0.9)]
        for stacking_index in (0, 1, 2):
            for px, py in points:
                v0 = layer_potential(
                    np.array([px]), np.array([py]), a, stacking_index=stacking_index
                )
                v1 = layer_potential(
                    np.array([px + shift[0]]),
                    np.array([py + shift[1]]),
                    a,
                    stacking_index=stacking_index,
                )
                assert v1[0, 0] == pytest.approx(v0[0, 0], abs=1e-9)


class TestStackPattern:
    def test_shape_range_finite(self):
        result = generate_stack_pattern(grid_size=64)
        assert result["pattern"].shape == (64, 64)
        assert result["pattern"].min() >= 0.0
        assert result["pattern"].max() <= 1.0
        assert np.all(np.isfinite(result["pattern"]))
        assert result["n_layers"] == 2
        assert len(result["x"]) == 64
        assert len(result["y"]) == 64

    def test_aa_ab_differ(self):
        p_aa = generate_stack_pattern("AA", grid_size=128, physical_extent=15.0)
        p_ab = generate_stack_pattern("AB", grid_size=128, physical_extent=15.0)
        assert np.max(np.abs(p_aa["pattern"] - p_ab["pattern"])) > 0.05

    def test_aba_abc_differ(self):
        p_aba = generate_stack_pattern("ABA", grid_size=128, physical_extent=15.0)
        p_abc = generate_stack_pattern("ABC", grid_size=128, physical_extent=15.0)
        assert p_aba["n_layers"] == 3
        assert p_abc["n_layers"] == 3
        assert np.max(np.abs(p_aba["pattern"] - p_abc["pattern"])) > 0.05

    def test_twisted_bilayer_period(self):
        result = generate_stack_pattern("twisted_bilayer", 1.08, grid_size=32)
        expected = 2.46 / (2.0 * np.sin(np.radians(1.08) / 2.0))
        assert result["moire_period"] == pytest.approx(expected, rel=1e-9)

    @pytest.mark.parametrize("stacking", ["AA", "AB", "ABA", "ABC"])
    def test_untwisted_periods_infinite(self, stacking):
        result = generate_stack_pattern(stacking, grid_size=32, physical_extent=15.0)
        assert np.isinf(result["moire_period"])

    def test_twist_sign_mirror(self):
        """Reversing the twist sign mirrors the pattern about the x axis."""
        p_neg = generate_stack_pattern("twisted_bilayer", -1.08, grid_size=64)
        p_pos = generate_stack_pattern("twisted_bilayer", 1.08, grid_size=64)
        np.testing.assert_allclose(p_neg["pattern"], p_pos["pattern"][::-1, :], atol=1e-10)

    def test_alternating_trilayer_period(self):
        result = generate_stack_pattern("alternating_trilayer", 1.5, grid_size=32)
        expected = 2.46 / (2.0 * np.sin(np.radians(1.5) / 2.0))
        assert result["moire_period"] == pytest.approx(expected, rel=1e-9)

    def test_invalid_grid_size_raises(self):
        with pytest.raises(ValueError):
            generate_stack_pattern(grid_size=1)

    def test_invalid_extent_raises(self):
        with pytest.raises(ValueError):
            generate_stack_pattern(physical_extent=0.0)

    def test_invalid_lattice_a_raises(self):
        with pytest.raises(ValueError):
            generate_stack_pattern(lattice_a=0.0)

    def test_invalid_stacking_raises(self):
        with pytest.raises(ValueError):
            generate_stack_pattern("BBB")


class TestHoneycombPotential:
    def test_origin_value(self):
        """A sublattice: 6; B sublattice: 4*cos(2*pi/3) + 2*cos(0) = 0; total 6."""
        v = layer_potential_honeycomb(np.array([0.0]), np.array([0.0]), stacking_index=0)
        assert v[0, 0] == pytest.approx(6.0, abs=1e-12)

    def test_honeycomb_pattern_normalized_and_differs(self):
        p_honeycomb = generate_stack_pattern_v2(
            "AB", grid_size=128, physical_extent=15.0, honeycomb=True
        )
        p_bravais = generate_stack_pattern_v2("AB", grid_size=128, physical_extent=15.0)
        assert np.all(np.isfinite(p_honeycomb["pattern"]))
        assert p_honeycomb["pattern"].min() >= 0.0
        assert p_honeycomb["pattern"].max() <= 1.0
        assert np.max(np.abs(p_honeycomb["pattern"] - p_bravais["pattern"])) > 0.05


class TestHeterostrain:
    def test_zero_strain_returns_input_exactly(self):
        g = _reciprocal_g_vectors_hexagonal(2.46)
        np.testing.assert_array_equal(strained_g_vectors(g, 0.0, 37.0), g)

    def test_strain_makes_untwisted_aa_moire_finite(self):
        """1% heterostrain turns the infinite-period aligned AA stack into a moire."""
        unstrained = generate_stack_pattern_v2(
            "AA", twist_angle_deg=0.0, grid_size=64, physical_extent=15.0
        )
        strained = generate_stack_pattern_v2(
            "AA", twist_angle_deg=0.0, grid_size=64, physical_extent=15.0,
            heterostrain_percent=1.0,
        )
        assert np.isinf(unstrained["moire_period"])
        assert np.isfinite(strained["moire_period"])
        assert strained["moire_period"] > 0.0
        assert np.max(np.abs(unstrained["pattern"] - strained["pattern"])) > 0.01

    def test_period_from_shells_twist_only(self):
        """Twist-only shells reduce exactly to a/(2*sin(theta/2))."""
        g = _reciprocal_g_vectors_hexagonal(2.46)
        for theta in (0.5, 1.08, 3.0):
            period = moire_period_from_g_shells(g, apply_rotation(g, theta))
            expected = 2.46 / (2.0 * np.sin(np.radians(theta) / 2.0))
            assert period == pytest.approx(expected, rel=1e-9)

    def test_mismatched_shell_shapes_raise(self):
        g = _reciprocal_g_vectors_hexagonal(2.46)
        with pytest.raises(ValueError, match="same shape"):
            moire_period_from_g_shells(g, g[:4])


class TestStackPatternV2:
    def test_defaults_bit_match_v1(self):
        v1 = generate_stack_pattern()
        v2 = generate_stack_pattern_v2()
        np.testing.assert_allclose(v2["pattern"], v1["pattern"], atol=0)
        np.testing.assert_allclose(v2["x"], v1["x"], atol=0)
        np.testing.assert_allclose(v2["y"], v1["y"], atol=0)
        assert v2["moire_period"] == v1["moire_period"]
        assert v2["n_layers"] == v1["n_layers"]

    def test_zero_displacement_bit_matches(self):
        base = generate_stack_pattern_v2(grid_size=64)
        warped = generate_stack_pattern_v2(
            grid_size=64, displacement=(np.zeros((64, 64)), np.zeros((64, 64)))
        )
        np.testing.assert_allclose(warped["pattern"], base["pattern"], atol=0)

    def test_bump_displacement_changes_pattern(self):
        base = generate_stack_pattern_v2(grid_size=64)
        bump = CurvatureConfig(geometry="gaussian_bump", amplitude=20.0, sigma=60.0)
        height = height_field(bump, base["x"], base["y"])
        warp = displacement_field(height, base["x"], base["y"])
        warped = generate_stack_pattern_v2(grid_size=64, displacement=warp)
        assert np.all(np.isfinite(warped["pattern"]))
        assert np.max(np.abs(warped["pattern"] - base["pattern"])) > 0.01

    def test_bad_displacement_shape_raises(self):
        with pytest.raises(ValueError, match="displacement"):
            generate_stack_pattern_v2(
                grid_size=64, displacement=(np.zeros((32, 32)), np.zeros((32, 32)))
            )


class TestSupermoire:
    def test_pattern_and_keys(self):
        result = generate_supermoire_pattern(grid_size=64)
        for key in (
            "x", "y", "pattern", "stack_period", "interface_period",
            "supermoire_period", "n_layers",
        ):
            assert key in result
        assert result["pattern"].shape == (64, 64)
        assert np.all(np.isfinite(result["pattern"]))
        assert result["pattern"].min() >= 0.0
        assert result["pattern"].max() <= 1.0
        assert result["n_layers"] == 3  # twisted bilayer + overlayer

    def test_aligned_graphene_overlayer_infinite(self):
        """Graphene-on-graphene, aligned: no interface moire, no supermoire."""
        result = generate_supermoire_pattern(
            overlayer_a=2.46, interface_twist_deg=0.0, grid_size=32
        )
        assert np.isinf(result["interface_period"])
        assert np.isinf(result["supermoire_period"])

    def test_sb2te3_overlayer_finite(self):
        """TBG at 1.08 deg on Sb2Te3 (a = 4.264, hexagonal): finite supermoire."""
        result = generate_supermoire_pattern(
            stacking="twisted_bilayer",
            twist_angle_deg=1.08,
            overlayer_a=4.264,
            overlayer_lattice_type="hexagonal",
            interface_twist_deg=0.0,
            grid_size=32,
        )
        assert np.isfinite(result["stack_period"])
        # Aligned hex-hex shells reduce to the 1D mismatch formula
        assert result["interface_period"] == pytest.approx(
            2.46 * 4.264 / (4.264 - 2.46), rel=1e-9
        )
        assert np.isfinite(result["supermoire_period"])
        assert result["supermoire_period"] > 0.0

    def test_invalid_overlayer_raises(self):
        with pytest.raises(ValueError):
            generate_supermoire_pattern(overlayer_a=0.0, grid_size=32)
        with pytest.raises(ValueError):
            generate_supermoire_pattern(overlayer_lattice_type="oblique", grid_size=32)


class TestFlatBand:
    def test_magic_angle_bilayer(self):
        theta = magic_angle_deg()
        assert theta == pytest.approx(1.076, abs=0.01)
        assert 1.0 < theta < 1.2

    def test_magic_angle_trilayer_ratio(self):
        """Khalaf-Vishwanath: trilayer magic angle is sqrt(2) x bilayer."""
        assert magic_angle_deg(3) / magic_angle_deg(2) == pytest.approx(
            np.sqrt(2.0), rel=1e-3
        )

    def test_velocity_zero_at_magic_angle(self):
        assert dirac_velocity_ratio(magic_angle_deg()) == pytest.approx(0.0, abs=1e-9)

    def test_velocity_half_at_twice_magic(self):
        assert dirac_velocity_ratio(2.15) == pytest.approx(0.5, abs=0.05)

    def test_velocity_near_unity_at_large_angle(self):
        assert dirac_velocity_ratio(30.0) > 0.99

    def test_velocity_negative_below_magic(self):
        assert dirac_velocity_ratio(0.8) < 0

    def test_velocity_vectorized(self):
        result = dirac_velocity_ratio(np.array([1.5, 2.15, 30.0]))
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)

    def test_velocity_nonpositive_theta_raises(self):
        with pytest.raises(ValueError):
            dirac_velocity_ratio(0.0)
        with pytest.raises(ValueError):
            dirac_velocity_ratio(-1.08)

    def test_magic_angle_invalid_layers_raises(self):
        with pytest.raises(ValueError):
            magic_angle_deg(4)


@pytest.mark.speculative
class TestFlatBandSC:
    def test_gap_peaks_at_magic_angle(self):
        theta_magic = magic_angle_deg()
        peak = compute_flat_band_sc(theta_magic, filling=2.4)
        assert peak.delta_mev == pytest.approx(DELTA_TBG_MAX, abs=1e-12)
        below = compute_flat_band_sc(theta_magic - 0.2, filling=2.4)
        above = compute_flat_band_sc(theta_magic + 0.2, filling=2.4)
        assert below.delta_mev < peak.delta_mev
        assert above.delta_mev < peak.delta_mev

    def test_tc_bilayer_peak(self):
        result = compute_flat_band_sc(magic_angle_deg(), filling=2.4)
        assert result.tc_kelvin == pytest.approx(1.974, abs=0.02)

    def test_tc_trilayer_peak(self):
        result = compute_flat_band_sc(magic_angle_deg(3), n_layers=3, filling=2.4)
        assert result.tc_kelvin == pytest.approx(2.895, abs=0.02)

    def test_filling_dome(self):
        assert filling_dome_factor(0.0) == 0.0
        assert filling_dome_factor(4.0) == 0.0
        assert filling_dome_factor(-2.4) == pytest.approx(1.0)

    def test_filling_out_of_range_raises(self):
        with pytest.raises(ValueError):
            filling_dome_factor(5.0)
        with pytest.raises(ValueError):
            compute_flat_band_sc(1.08, filling=5.0)

    def test_invalid_n_layers_raises(self):
        with pytest.raises(ValueError):
            compute_flat_band_sc(1.08, n_layers=4)

    def test_cpdm_washout_at_tbg_coherence_length(self):
        """xi ~ 500 A >> L_m ~ 130 A: CPDM contrast is tiny, as expected."""
        assert cpdm_amplitude(130.51, coherence_length=XI_TBG) == pytest.approx(
            0.022, abs=0.001
        )

    def test_gap_modulation_on_stack_pattern(self):
        result = generate_stack_pattern("twisted_bilayer", 1.08, grid_size=64)
        gap = gap_modulation(result["pattern"], 0.30, 0.15)
        assert np.all(np.isfinite(gap))
        assert np.all(gap >= 0.15)
        assert np.all(gap <= 0.45)
