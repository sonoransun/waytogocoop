"""Tests for curved-sheet strain and pseudo-magnetic-field computation."""

from __future__ import annotations

import numpy as np
import pytest

from waytogocoop.computation.curvature import (
    CurvatureConfig,
    compute_curvature_effects,
    displacement_field,
    gap_suppression_factor,
    height_field,
    strain_tensor,
)

GRID = np.linspace(-100.0, 100.0, 200)


def _field(config: CurvatureConfig) -> np.ndarray:
    return compute_curvature_effects(config, GRID, GRID).pseudo_field


class TestHeightFields:
    def test_flat_all_zero(self):
        h = height_field(CurvatureConfig(geometry="flat"), GRID, GRID)
        assert np.all(h == 0.0)

    def test_bump_peak_equals_amplitude(self):
        config = CurvatureConfig(geometry="gaussian_bump", amplitude=5.0, sigma=50.0)
        h = height_field(config, np.array([0.0]), np.array([0.0]))
        assert h[0, 0] == pytest.approx(5.0, rel=1e-12)

    def test_bump_decays_beyond_four_sigma(self):
        config = CurvatureConfig(geometry="gaussian_bump", amplitude=5.0, sigma=20.0)
        h = height_field(config, GRID, GRID)
        X, Y = np.meshgrid(GRID, GRID)
        far = np.sqrt(X**2 + Y**2) > 4.0 * config.sigma
        assert h[far].max() < 0.01 * config.amplitude

    def test_ripple_amplitude(self):
        config = CurvatureConfig(
            geometry="sinusoidal_ripple", amplitude=2.0, wavelength=100.0
        )
        h = height_field(config, GRID, GRID)
        assert h.max() == pytest.approx(2.0, rel=1e-2)

    def test_cylinder_constant_along_y(self):
        config = CurvatureConfig(
            geometry="cylindrical_bend", radius=1000.0, orientation_deg=0.0
        )
        h = height_field(config, GRID, GRID)
        np.testing.assert_array_equal(h, np.broadcast_to(h[0], h.shape))

    def test_cap_max_at_center_and_nonnegative(self):
        config = CurvatureConfig(geometry="spherical_cap", radius=2000.0)
        h = height_field(config, GRID, GRID)
        assert np.all(h >= 0.0)
        assert h.max() == pytest.approx(2000.0, rel=1e-6)
        iy, ix = np.unravel_index(np.argmax(h), h.shape)
        assert iy in (99, 100)
        assert ix in (99, 100)

    def test_unknown_geometry_raises(self):
        with pytest.raises(ValueError, match="Unknown geometry"):
            height_field(CurvatureConfig(geometry="dome"), GRID, GRID)


class TestStrain:
    def test_flat_zero_strain(self):
        h = height_field(CurvatureConfig(geometry="flat"), GRID, GRID)
        eps_xx, eps_yy, eps_xy = strain_tensor(h, GRID, GRID)
        assert np.all(eps_xx == 0.0)
        assert np.all(eps_yy == 0.0)
        assert np.all(eps_xy == 0.0)

    def test_bump_strain_small_and_positive(self):
        h = height_field(CurvatureConfig(geometry="gaussian_bump"), GRID, GRID)
        eps_xx, _, _ = strain_tensor(h, GRID, GRID)
        assert 0.0 < eps_xx.max() < 0.05

    def test_ripple_zigzag_pure_xx(self):
        """A ripple along x (zigzag) has hy = 0 exactly, so eps_yy = eps_xy = 0."""
        config = CurvatureConfig(
            geometry="sinusoidal_ripple", amplitude=2.0, wavelength=100.0,
            orientation_deg=0.0,
        )
        h = height_field(config, GRID, GRID)
        _, eps_yy, eps_xy = strain_tensor(h, GRID, GRID)
        assert np.all(eps_yy == 0.0)
        assert np.all(eps_xy == 0.0)


class TestPseudoField:
    def test_flat_zero_field(self):
        assert np.all(_field(CurvatureConfig(geometry="flat")) == 0.0)

    def test_ripple_along_zigzag_gives_zero(self):
        config = CurvatureConfig(
            geometry="sinusoidal_ripple", amplitude=2.0, wavelength=100.0,
            orientation_deg=0.0,
        )
        assert np.max(np.abs(_field(config))) < 1e-8

    @pytest.mark.parametrize("phi_deg", [10.0, 20.0])
    def test_ripple_sin_3phi_law(self, phi_deg):
        def ripple_max(phi: float) -> float:
            config = CurvatureConfig(
                geometry="sinusoidal_ripple", amplitude=2.0, wavelength=100.0,
                orientation_deg=phi,
            )
            return float(np.max(np.abs(_field(config))))

        ratio = ripple_max(phi_deg) / ripple_max(30.0)
        assert ratio == pytest.approx(np.sin(np.radians(3.0 * phi_deg)), rel=0.02)

    def test_ripple_magnitude(self):
        """Analytic: max|B| = PREF * h0^2 * k^3 * |sin(3*phi)| / 2 -> 34.5 T."""
        config = CurvatureConfig(
            geometry="sinusoidal_ripple", amplitude=2.0, wavelength=100.0,
            orientation_deg=30.0,
        )
        assert np.max(np.abs(_field(config))) == pytest.approx(34.49, rel=0.05)

    def test_cylinder_along_zigzag_gives_zero(self):
        config = CurvatureConfig(
            geometry="cylindrical_bend", radius=1000.0, orientation_deg=0.0
        )
        assert np.max(np.abs(_field(config))) < 1e-8

    def test_cylinder_tilted_gives_field(self):
        config = CurvatureConfig(
            geometry="cylindrical_bend", radius=1000.0, orientation_deg=30.0
        )
        assert np.max(np.abs(_field(config))) > 1.0

    def test_bump_magnitude(self):
        config = CurvatureConfig(geometry="gaussian_bump", amplitude=5.0, sigma=50.0)
        assert np.max(np.abs(_field(config))) == pytest.approx(5.70, rel=0.05)

    def test_bump_zigzag_node(self):
        """B ~ sin(3*theta): the field vanishes along the zigzag (y = 0) axis."""
        field = _field(CurvatureConfig(geometry="gaussian_bump", amplitude=5.0, sigma=50.0))
        row = np.argmin(np.abs(GRID))
        assert np.max(np.abs(field[row, :])) < 0.05 * np.max(np.abs(field))

    def test_cap_field_small(self):
        bump_max = np.max(
            np.abs(_field(CurvatureConfig(geometry="gaussian_bump", amplitude=5.0, sigma=50.0)))
        )
        cap_max = np.max(np.abs(_field(CurvatureConfig(geometry="spherical_cap", radius=2000.0))))
        assert 0.0 < cap_max < 0.05 * bump_max

    def test_bump_zero_net_flux(self):
        field = _field(CurvatureConfig(geometry="gaussian_bump", amplitude=5.0, sigma=50.0))
        assert np.abs(field.mean()) < 0.02 * np.max(np.abs(field))


class TestValley:
    def test_valley_flip_negates_field_exactly(self):
        field_k = _field(
            CurvatureConfig(geometry="gaussian_bump", amplitude=5.0, sigma=50.0, valley=1)
        )
        field_kp = _field(
            CurvatureConfig(geometry="gaussian_bump", amplitude=5.0, sigma=50.0, valley=-1)
        )
        np.testing.assert_array_equal(field_kp, -1 * field_k)

    def test_invalid_valley_raises(self):
        with pytest.raises(ValueError, match="valley"):
            compute_curvature_effects(CurvatureConfig(valley=0), GRID, GRID)

    @pytest.mark.speculative
    def test_gap_suppression_valley_independent(self):
        """Gap suppression uses |B|, so both valleys give the same factor."""
        result_k = compute_curvature_effects(
            CurvatureConfig(geometry="gaussian_bump", valley=1), GRID, GRID
        )
        result_kp = compute_curvature_effects(
            CurvatureConfig(geometry="gaussian_bump", valley=-1), GRID, GRID
        )
        np.testing.assert_array_equal(result_k.gap_suppression, result_kp.gap_suppression)


@pytest.mark.speculative
class TestDisplacementField:
    def test_flat_gives_zero_displacement(self):
        u_x, u_y = displacement_field(np.zeros((200, 200)), GRID, GRID)
        assert np.all(u_x == 0.0)
        assert np.all(u_y == 0.0)

    def test_bump_displacement_sign_and_scale(self):
        """u = -h*grad(h)/2 points down the gradient; |u(r=sigma)| ~ h0^2/(2*sigma)."""
        amplitude, sigma = 5.0, 50.0
        config = CurvatureConfig(geometry="gaussian_bump", amplitude=amplitude, sigma=sigma)
        height = height_field(config, GRID, GRID)
        u_x, _ = displacement_field(height, GRID, GRID)
        iy = np.argmin(np.abs(GRID))            # y ~ 0 row
        ix_pos = np.argmin(np.abs(GRID - sigma))  # x ~ +sigma
        ix_neg = np.argmin(np.abs(GRID + sigma))  # x ~ -sigma
        # h > 0 and dh/dx < 0 at x = +sigma, so u_x = -h*dh/dx/2 > 0 (and mirrored)
        assert u_x[iy, ix_pos] > 0.0
        assert u_x[iy, ix_neg] < 0.0
        scale = amplitude**2 / (2.0 * sigma)
        assert scale / 3.0 < abs(u_x[iy, ix_pos]) < 3.0 * scale


@pytest.mark.speculative
class TestGapSuppression:
    def test_zero_field_gives_unity(self):
        factor = gap_suppression_factor(np.zeros((5, 5)))
        assert np.all(factor == 1.0)

    def test_strong_field_gives_zero(self):
        assert np.all(gap_suppression_factor(np.full((3, 3), 12.0), b_pairbreak=10.0) == 0.0)
        assert np.all(gap_suppression_factor(np.full((3, 3), -12.0), b_pairbreak=10.0) == 0.0)

    def test_bounded(self):
        factor = gap_suppression_factor(np.linspace(-30.0, 30.0, 61).reshape((1, 61)))
        assert np.all(factor >= 0.0)
        assert np.all(factor <= 1.0)

    def test_orchestrator_shapes_and_max_field(self):
        config = CurvatureConfig(geometry="gaussian_bump", amplitude=5.0, sigma=50.0)
        result = compute_curvature_effects(config, GRID, GRID)
        for field in (
            result.height,
            result.strain_xx,
            result.strain_yy,
            result.strain_xy,
            result.pseudo_field,
            result.gap_suppression,
        ):
            assert field.shape == (200, 200)
            assert np.all(np.isfinite(field))
        assert result.max_abs_field == np.abs(result.pseudo_field).max()

    def test_orchestrator_flat(self):
        result = compute_curvature_effects(CurvatureConfig(geometry="flat"), GRID, GRID)
        assert np.all(result.gap_suppression == 1.0)
        assert result.max_abs_field == 0.0

    @pytest.mark.parametrize(
        "config",
        [
            CurvatureConfig(geometry="dome"),
            CurvatureConfig(amplitude=-1.0),
            CurvatureConfig(sigma=0.0),
            CurvatureConfig(wavelength=0.0),
            CurvatureConfig(radius=0.0),
            CurvatureConfig(b_pairbreak=0.0),
        ],
    )
    def test_invalid_config_raises(self, config):
        with pytest.raises(ValueError):
            compute_curvature_effects(config, GRID, GRID)

    def test_too_few_grid_points_raises(self):
        with pytest.raises(ValueError):
            compute_curvature_effects(CurvatureConfig(), np.linspace(-1.0, 1.0, 4), GRID)
