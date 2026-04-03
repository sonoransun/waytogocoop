"""Tests for topological computation module."""

from __future__ import annotations

import numpy as np
import pytest

from waytogocoop.computation.topological import (
    ProximityConfig,
    proximity_decay_profile,
    gap_3d,
    dirac_dispersion,
    cooper_surface_3d,
    majorana_probability_density,
    topological_phase_index,
    phase_diagram_sweep,
    topological_magnetoelectric_polarization,
    chern_number_estimate,
)


class TestProximityDecay:
    def test_bulk_sc_unity(self):
        """For z < 0, f(z) = 1."""
        z = np.array([-50.0, -10.0, -1.0])
        profile = proximity_decay_profile(z)
        np.testing.assert_allclose(profile, 1.0)

    def test_interface_transparency(self):
        """At z = 0, f(z) = T."""
        z = np.array([0.0])
        profile = proximity_decay_profile(z, interface_transparency=0.8)
        assert profile[0] == pytest.approx(0.8, abs=1e-10)

    def test_monotonic_decay(self):
        """f(z) decreases monotonically for z > 0."""
        z = np.linspace(0, 500, 200)
        profile = proximity_decay_profile(z)
        diff = np.diff(profile)
        assert np.all(diff <= 1e-15), "Profile is not monotonically decreasing"

    def test_asymptotic_zero(self):
        """Far into the TI, f(z) → 0."""
        z = np.array([5000.0])
        profile = proximity_decay_profile(z, xi_prox=100.0)
        assert profile[0] < 1e-10


class TestGap3D:
    def test_shape(self):
        gap_2d = np.ones((50, 50)) * 3.0
        config = ProximityConfig(n_z_layers=20)
        result = gap_3d(gap_2d, config)
        assert result.gap_3d.shape == (20, 50, 50)
        assert len(result.z_coords) == 20
        assert len(result.decay_profile) == 20

    def test_bulk_sc_preserves_gap(self):
        """In the SC region (z < 0), gap should equal the 2D gap."""
        gap_2d = np.ones((10, 10)) * 3.0
        config = ProximityConfig(n_z_layers=10, z_min=-100.0, z_max=-10.0)
        result = gap_3d(gap_2d, config)
        # All z < 0, so f(z) = 1.0, gap_3d should equal gap_2d
        np.testing.assert_allclose(result.gap_3d, 3.0, atol=1e-10)

    def test_deep_ti_small_gap(self):
        """Deep in TI, gap should be strongly suppressed."""
        gap_2d = np.ones((10, 10)) * 3.0
        config = ProximityConfig(
            n_z_layers=10, z_min=400.0, z_max=500.0, xi_prox=100.0
        )
        result = gap_3d(gap_2d, config)
        assert result.gap_3d.max() < 0.1


class TestDiracDispersion:
    def test_linear(self):
        """E should be proportional to |k|."""
        e1 = dirac_dispersion(np.array([0.1]))
        e2 = dirac_dispersion(np.array([0.2]))
        ratio = e2[0] / e1[0]
        assert ratio == pytest.approx(2.0, rel=0.01)

    def test_zero_k(self):
        e = dirac_dispersion(np.array([0.0]))
        assert e[0] == 0.0

    def test_positive_energy(self):
        e = dirac_dispersion(np.array([0.1]))
        assert e[0] > 0


class TestCooperSurface3D:
    def test_shape(self):
        gap_2d = np.ones((20, 20)) * 3.0
        x = np.linspace(-50, 50, 20)
        y = np.linspace(-50, 50, 20)
        config = ProximityConfig(n_z_layers=10)
        result = cooper_surface_3d(gap_2d, x, y, moire_period=100.0, proximity_config=config)
        assert result.field_3d.shape == (10, 20, 20)
        assert len(result.z) == 10

    def test_with_suppression(self):
        gap_2d = np.ones((10, 10)) * 3.0
        supp = np.ones((10, 10)) * 0.5
        x = np.linspace(-50, 50, 10)
        config = ProximityConfig(n_z_layers=5, z_min=-10.0, z_max=-1.0)
        result = cooper_surface_3d(
            gap_2d, x, x, moire_period=100.0,
            suppression_field=supp, proximity_config=config,
        )
        # In bulk SC (z < 0), gap = 3 * 0.5 = 1.5
        np.testing.assert_allclose(result.field_3d, 1.5, atol=1e-10)


class TestMajorana:
    """SPECULATIVE — tests verify internal consistency, not physical accuracy."""

    def test_no_vortices_no_density(self):
        x = np.linspace(-50, 50, 20)
        pos = np.empty((0, 2))
        result = majorana_probability_density(x, x, pos)
        np.testing.assert_allclose(result.probability_density, 0.0)
        assert result.n_vortices_with_mzm == 0

    def test_non_negative(self):
        x = np.linspace(-100, 100, 50)
        pos = np.array([[0.0, 0.0]])
        result = majorana_probability_density(x, x, pos)
        assert np.all(result.probability_density >= 0)

    def test_peaks_at_vortex_core(self):
        """Density should be maximal near the vortex core."""
        x = np.linspace(-100, 100, 101)
        pos = np.array([[0.0, 0.0]])
        result = majorana_probability_density(x, x, pos)
        center_val = result.probability_density[50, 50]
        # Center should be the peak (normalized to 1)
        assert center_val == pytest.approx(1.0, rel=0.1)

    def test_localization_length_stored(self):
        x = np.linspace(-50, 50, 10)
        pos = np.array([[0.0, 0.0]])
        result = majorana_probability_density(x, x, pos, xi_M=75.0)
        assert result.localization_length == 75.0


class TestTopologicalPhase:
    """SPECULATIVE — simplified Fu-Kane criterion."""

    def test_trivial_phase(self):
        """E_Z < Delta → trivial."""
        assert topological_phase_index(1.0, 3.0) == 0

    def test_topological_phase(self):
        """E_Z > Delta → topological."""
        assert topological_phase_index(5.0, 3.0) == 1

    def test_with_chemical_potential(self):
        """E_Z > sqrt(Delta^2 + mu^2) needed for topological."""
        # E_Z = 4 meV, Delta = 3 meV, mu = 2 meV
        # sqrt(9 + 4) = 3.606, so 4 > 3.606 → topological
        assert topological_phase_index(4.0, 3.0, 2.0) == 1
        # E_Z = 3 meV → 3 < 3.606 → trivial
        assert topological_phase_index(3.0, 3.0, 2.0) == 0

    def test_phase_diagram_sweep_shape(self):
        B_values = np.linspace(0, 100, 10)
        delta_values = np.linspace(0.1, 10, 10)
        phase = phase_diagram_sweep(B_values, delta_values)
        assert phase.shape == (10, 10)

    def test_phase_diagram_has_both_phases(self):
        """Wide enough sweep should contain both 0 and 1."""
        B_values = np.linspace(0, 200, 50)
        delta_values = np.linspace(0.1, 10, 50)
        phase = phase_diagram_sweep(B_values, delta_values)
        assert 0 in phase
        assert 1 in phase


class TestMagnetoelectric:
    """SPECULATIVE — topological magnetoelectric effect."""

    def test_positive_for_positive_B(self):
        P = topological_magnetoelectric_polarization(1.0)
        assert P > 0

    def test_zero_for_zero_B(self):
        P = topological_magnetoelectric_polarization(0.0)
        assert P == 0.0


class TestChernNumber:
    def test_topological_half(self):
        C = chern_number_estimate(3.0, 5.0)  # E_Z > Delta
        assert C == pytest.approx(0.5)

    def test_trivial_negative_half(self):
        C = chern_number_estimate(3.0, 1.0)  # E_Z < Delta
        assert C == pytest.approx(-0.5)

    def test_at_boundary(self):
        C = chern_number_estimate(3.0, 3.0)  # E_Z = Delta
        assert C == 0.0
