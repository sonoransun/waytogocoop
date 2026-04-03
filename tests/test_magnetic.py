"""Tests for magnetic field computation module."""

from __future__ import annotations

import numpy as np
import pytest

from waytogocoop.computation.magnetic import (
    MagneticFieldConfig,
    vortex_lattice_period,
    commensuration_field,
    generate_vortex_positions,
    vortex_suppression_field,
    combined_gap_with_vortices,
    flux_per_moire_cell,
    zeeman_energy,
    pauli_limiting_field,
    compute_zeeman,
    screening_currents,
    local_susceptibility,
    moire_vortex_beating,
    field_tunable_cpdm,
    commensuration_pinning_energy,
)
from waytogocoop.config import PHI_0, ANGSTROM_TO_M


class TestVortexLattice:
    def test_period_at_1_tesla(self):
        """a_v at 1T should be ~489 Angstrom."""
        p = vortex_lattice_period(1.0)
        assert 480 < p < 500, f"period = {p}"

    def test_period_scales_inverse_sqrt_B(self):
        """a_v(4T) = a_v(1T) / 2."""
        p1 = vortex_lattice_period(1.0)
        p4 = vortex_lattice_period(4.0)
        assert p4 == pytest.approx(p1 / 2.0, rel=0.01)

    def test_zero_field_infinite_period(self):
        assert vortex_lattice_period(0.0) == float("inf")

    def test_negative_field(self):
        """Negative Bz should give same period as positive."""
        assert vortex_lattice_period(-1.0) == pytest.approx(
            vortex_lattice_period(1.0), rel=1e-10
        )

    def test_commensuration_field_sb2te3(self):
        """For L_m = 36.7 A, B_comm should be very large."""
        B = commensuration_field(36.7)
        assert B > 100, f"B_comm = {B}"

    def test_commensuration_field_infinite_period(self):
        assert commensuration_field(float("inf")) == float("inf")


class TestVortexPositions:
    def test_zero_field_empty(self):
        pos = generate_vortex_positions(0.0, 100.0, 200)
        assert pos.shape == (0, 2)

    def test_nonempty_at_1T(self):
        pos = generate_vortex_positions(1.0, 1000.0, 200)
        assert len(pos) > 0

    def test_within_viewport(self):
        extent = 500.0
        pos = generate_vortex_positions(1.0, extent, 200)
        half = extent / 2.0
        assert np.all(pos[:, 0] >= -half)
        assert np.all(pos[:, 0] <= half)
        assert np.all(pos[:, 1] >= -half)
        assert np.all(pos[:, 1] <= half)


class TestVortexSuppression:
    def test_no_vortices_all_ones(self):
        x = np.linspace(-50, 50, 100)
        pos = np.empty((0, 2))
        supp = vortex_suppression_field(x, x, pos)
        np.testing.assert_allclose(supp, 1.0)

    def test_suppression_near_core(self):
        """At a vortex core, suppression should approach 0."""
        x = np.linspace(-100, 100, 201)
        pos = np.array([[0.0, 0.0]])
        supp = vortex_suppression_field(x, x, pos, coherence_length=20.0)
        center = supp[100, 100]
        assert center < 0.05, f"center suppression = {center}"

    def test_suppression_far_from_core(self):
        """Far from all cores, suppression ≈ 1."""
        x = np.linspace(-1000, 1000, 200)
        pos = np.array([[0.0, 0.0]])
        supp = vortex_suppression_field(x, x, pos, coherence_length=20.0)
        corner = supp[0, 0]
        assert corner > 0.99, f"corner suppression = {corner}"


class TestCombinedGap:
    def test_shape_preserved(self):
        gap = np.ones((100, 100)) * 3.0
        supp = np.ones((100, 100)) * 0.5
        combined = combined_gap_with_vortices(gap, supp)
        assert combined.shape == (100, 100)
        np.testing.assert_allclose(combined, 1.5)

    def test_zero_suppression(self):
        gap = np.ones((10, 10)) * 3.0
        supp = np.zeros((10, 10))
        combined = combined_gap_with_vortices(gap, supp)
        np.testing.assert_allclose(combined, 0.0)


class TestFluxQuantization:
    def test_integer_flux(self):
        """Compute B giving exactly 1 flux quantum per moire cell."""
        L_m = 100.0  # Angstrom
        A_m = np.sqrt(3) / 2.0 * (L_m * ANGSTROM_TO_M) ** 2
        B = PHI_0 / A_m
        n = flux_per_moire_cell(B, L_m)
        assert n == pytest.approx(1.0, rel=0.01)

    def test_zero_field_zero_flux(self):
        assert flux_per_moire_cell(0.0, 100.0) == 0.0


class TestZeeman:
    def test_zero_field_zero_energy(self):
        assert zeeman_energy(0.0, 0.0) == 0.0

    def test_positive_energy(self):
        E = zeeman_energy(1.0, 0.0)
        assert E > 0

    def test_pauli_limit_reasonable(self):
        """For Delta = 3 meV, B_P ~ 37 T."""
        B_P = pauli_limiting_field(3.0)
        assert 30 < B_P < 50, f"B_P = {B_P}"

    def test_compute_zeeman_full(self):
        config = MagneticFieldConfig(Bx=1.0, By=0.0, Bz=0.0)
        result = compute_zeeman(config, 3.0)
        assert result.zeeman_energy > 0
        assert result.pauli_limit_field > 0
        assert result.depairing_ratio >= 0


class TestScreeningCurrents:
    def test_no_vortices_zero_current(self):
        x = np.linspace(-50, 50, 20)
        pos = np.empty((0, 2))
        Jx, Jy = screening_currents(x, x, pos)
        np.testing.assert_allclose(Jx, 0.0)
        np.testing.assert_allclose(Jy, 0.0)

    def test_nonzero_with_vortex(self):
        x = np.linspace(-100, 100, 50)
        pos = np.array([[0.0, 0.0]])
        Jx, Jy = screening_currents(x, x, pos)
        J_mag = np.sqrt(Jx**2 + Jy**2)
        assert J_mag.max() == pytest.approx(1.0, rel=0.01)


class TestLocalSusceptibility:
    def test_uniform_gap(self):
        """Uniform gap → chi = 0 everywhere."""
        gap = np.ones((10, 10)) * 3.0
        chi = local_susceptibility(gap)
        np.testing.assert_allclose(chi, 0.0, atol=1e-10)

    def test_zero_gap_region(self):
        """Where gap = 0, chi should be -1."""
        gap = np.zeros((10, 10))
        gap[5, 5] = 1.0  # one point nonzero to set scale
        chi = local_susceptibility(gap)
        assert chi[0, 0] == pytest.approx(-1.0, abs=1e-10)


class TestSpeculativePhysics:
    def test_beating_commensurate(self):
        """Equal periods → uniform pattern."""
        x = np.linspace(-50, 50, 20)
        pattern = moire_vortex_beating(100.0, 100.0, x, x)
        np.testing.assert_allclose(pattern, 1.0)

    def test_beating_normalized(self):
        x = np.linspace(-100, 100, 50)
        pattern = moire_vortex_beating(100.0, 80.0, x, x)
        assert pattern.min() >= -1e-10
        assert pattern.max() <= 1.0 + 1e-10

    def test_field_tunable_cpdm_zero_field(self):
        A = field_tunable_cpdm(100.0, 20.0, 0.0)
        # Should equal the base CPDM amplitude
        from waytogocoop.computation.superconducting import cpdm_amplitude

        A0 = cpdm_amplitude(100.0, 20.0)
        assert A == pytest.approx(A0, rel=1e-10)

    def test_field_tunable_cpdm_at_bc2(self):
        A = field_tunable_cpdm(100.0, 20.0, 47.0, Bc2=47.0)
        assert A == pytest.approx(0.0, abs=1e-10)

    def test_pinning_commensurate(self):
        """L_m = L_v → cos(2pi) = 1."""
        E = commensuration_pinning_energy(100.0, 100.0)
        assert E == pytest.approx(1.0, abs=1e-10)

    def test_pinning_anticommensurate(self):
        """L_m = L_v/2 → cos(4pi) = 1."""
        E = commensuration_pinning_energy(100.0, 50.0)
        assert E == pytest.approx(1.0, abs=1e-10)
