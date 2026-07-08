"""Tests for the Bistritzer-MacDonald continuum model of twisted bilayer graphene."""

from __future__ import annotations

import numpy as np
import pytest

from waytogocoop.computation.bm_model import (
    BMConfig,
    build_bm_hamiltonian,
    compute_band_structure,
    compute_dos,
    flat_band_width_mev,
)
from waytogocoop.computation.graphene import dirac_velocity_ratio
from waytogocoop.config import GRAPHENE_A, HBAR_VF_GRAPHENE


def _k_theta(twist_angle_deg: float, lattice_a: float = GRAPHENE_A) -> float:
    k_dirac = 4.0 * np.pi / (3.0 * lattice_a)
    return 2.0 * k_dirac * np.sin(np.radians(twist_angle_deg) / 2.0)


class TestHamiltonian:
    def test_hamiltonian_hermitian(self):
        h = build_bm_hamiltonian(0.013, -0.007, BMConfig())
        assert np.max(np.abs(h - h.conj().T)) < 1e-12

    def test_hamiltonian_dim(self):
        """dim = 2*(sites_l1 + sites_l2) with 3*N^2 + 3*N + 1 sites per layer."""
        dims = []
        for n_shells in (1, 2, 3):
            h = build_bm_hamiltonian(0.0, 0.0, BMConfig(n_shells=n_shells))
            n_sites = 3 * n_shells**2 + 3 * n_shells + 1
            assert h.shape == (2 * (n_sites + n_sites), 2 * (n_sites + n_sites))
            assert h.shape[0] % 2 == 0
            dims.append(h.shape[0])
        assert dims[0] < dims[1] < dims[2]

    def test_dirac_point_degeneracy(self):
        """The layer-1 Dirac point at K_m stays within a few meV of zero."""
        energies_mev = 1.0e3 * np.linalg.eigvalsh(build_bm_hamiltonian(0.0, 0.0, BMConfig()))
        assert np.min(np.abs(energies_mev)) < 5.0

    def test_layer2_dirac_cone_at_k_prime_tick(self):
        """The K' path corner at -q1 = (0, +k_theta) carries the layer-2 Dirac cone.

        Pins the path-label orientation: the mirror point (0, -k_theta) is not
        a Dirac point in this convention and sits much further from zero.
        """
        config = BMConfig()
        k_theta = _k_theta(config.twist_angle_deg)
        at_k_prime = 1.0e3 * np.linalg.eigvalsh(build_bm_hamiltonian(0.0, k_theta, config))
        at_mirror = 1.0e3 * np.linalg.eigvalsh(build_bm_hamiltonian(0.0, -k_theta, config))
        assert np.min(np.abs(at_k_prime)) < 5.0
        assert np.min(np.abs(at_mirror)) > 2.0 * np.min(np.abs(at_k_prime))

    def test_chiral_limit_particle_hole_symmetry(self):
        """w_aa = 0: sublattice chiral symmetry forces an exactly symmetric spectrum.

        Pins the tunneling matrices — any phase error in T_j breaks this.
        """
        config = BMConfig(twist_angle_deg=1.08, w_aa=0.0)
        for kx, ky in [(0.011, 0.004), (0.0, -0.02), (-0.007, 0.013)]:
            energies_mev = 1.0e3 * np.sort(
                np.linalg.eigvalsh(build_bm_hamiltonian(kx, ky, config))
            )
            np.testing.assert_allclose(energies_mev, -energies_mev[::-1], atol=1e-6)

    def test_velocity_matches_first_order(self):
        """Numerical Dirac velocity vs the first-order (1-3a^2)/(1+6a^2) formula."""
        config = BMConfig(twist_angle_deg=2.5, w_aa=0.110, w_ab=0.110)
        k_theta = _k_theta(2.5)
        dk = 0.02 * k_theta
        direction = np.array([np.sqrt(3.0) / 2.0, -0.5])  # K -> Gamma
        e_at_k = np.linalg.eigvalsh(build_bm_hamiltonian(0.0, 0.0, config))
        e_at_dk = np.linalg.eigvalsh(
            build_bm_hamiltonian(dk * direction[0], dk * direction[1], config)
        )
        mid = len(e_at_k) // 2  # band just above zero
        v_ratio_numerical = (e_at_dk[mid] - e_at_k[mid]) / dk / HBAR_VF_GRAPHENE
        v_ratio_analytic = dirac_velocity_ratio(2.5)
        assert v_ratio_numerical == pytest.approx(v_ratio_analytic, rel=0.12)

    def test_valley_spectrum_symmetry(self):
        """K and K' valleys are time-reversal partners: same eigenvalue set at K_m."""
        e_plus = np.sort(np.linalg.eigvalsh(build_bm_hamiltonian(0.0, 0.0, BMConfig(valley=1))))
        e_minus = np.sort(np.linalg.eigvalsh(build_bm_hamiltonian(0.0, 0.0, BMConfig(valley=-1))))
        np.testing.assert_allclose(1.0e3 * e_plus, 1.0e3 * e_minus, atol=1e-9)

    @pytest.mark.parametrize(
        "config",
        [
            BMConfig(twist_angle_deg=0.0),
            BMConfig(twist_angle_deg=-1.08),
            BMConfig(valley=0),
            BMConfig(n_shells=0),
            BMConfig(w_aa=-0.01),
            BMConfig(w_ab=0.0),
        ],
    )
    def test_invalid_config_raises(self, config):
        with pytest.raises(ValueError):
            build_bm_hamiltonian(0.0, 0.0, config)


class TestBandStructure:
    def test_magic_angle_bandwidth_minimum(self):
        """Flat-band width is minimal near the magic angle, not at the scan edges."""
        thetas = [0.9, 1.0, 1.1, 1.2, 1.3]
        widths = [
            flat_band_width_mev(BMConfig(twist_angle_deg=theta), n_k_per_segment=6)
            for theta in thetas
        ]
        i_min = int(np.argmin(widths))
        assert 0 < i_min < len(thetas) - 1
        w_min = widths[i_min]
        assert w_min < 25.0
        assert flat_band_width_mev(BMConfig(twist_angle_deg=2.0)) > 4.0 * w_min

    def test_band_structure_shape_and_ticks(self):
        bs = compute_band_structure(BMConfig(), n_k_per_segment=6)
        n_k = 3 * 6 + 1
        assert bs.k_distances.shape == (n_k,)
        assert bs.energies_mev.shape[0] == n_k
        assert np.all(np.diff(bs.k_distances) >= 0.0)
        assert np.all(np.diff(bs.energies_mev, axis=1) >= 0.0)  # sorted per k
        assert bs.tick_labels == ["K", "Γ", "M", "K'"]
        assert len(bs.tick_positions) == 4
        assert bs.flat_bandwidth_mev > 0.0
        # Path geometry: adjacent hexagon corners are one circumradius k_theta
        # apart (K -> Gamma), the edge midpoint M is sqrt(3)/2*k_theta from
        # Gamma, and K' is k_theta/2 beyond M.
        k_theta = _k_theta(1.08)
        assert bs.tick_positions[0] == 0.0
        assert bs.tick_positions[1] == pytest.approx(k_theta, rel=1e-9)
        assert bs.tick_positions[2] - bs.tick_positions[1] == pytest.approx(
            np.sqrt(3.0) / 2.0 * k_theta, rel=1e-9
        )
        assert bs.tick_positions[3] - bs.tick_positions[2] == pytest.approx(
            k_theta / 2.0, rel=1e-9
        )


class TestDOS:
    def test_dos_properties(self):
        result = compute_dos(BMConfig(twist_angle_deg=1.08))
        energies = result["energies_mev"]
        dos = result["dos"]
        assert energies.shape == (200,)
        assert dos.shape == (200,)
        assert np.all(dos >= 0.0)
        assert np.abs(energies[np.argmax(dos)]) < 15.0  # flat-band peak near zero
        assert np.trapezoid(dos, energies) > 0.0
        # energy window respected
        assert energies.min() == -150.0
        assert energies.max() == 150.0
        narrow = compute_dos(BMConfig(), n_k_grid=4, e_window_mev=50.0, n_bins=64)
        assert narrow["energies_mev"].min() == -50.0
        assert narrow["energies_mev"].max() == 50.0
        assert len(narrow["energies_mev"]) == 64
