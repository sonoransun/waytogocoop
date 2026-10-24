"""Tests for the spin-fluctuation pairing module (SPECULATIVE).

LCCO magnetotransport of Duffy et al., Nat. Phys. (2026), doi:10.1038/s41567-026-03454-6
(page and figure numbers from arXiv:2502.13612 v1), plus the what-if pairing map and
moire glue modulation.  Paper values quoted here are read off the figures.
"""

from __future__ import annotations

import inspect
import math
import time
from dataclasses import replace
from decimal import Decimal
from functools import cache

import numpy as np
import pytest

from waytogocoop import config
from waytogocoop.computation import spin_fluctuation
from waytogocoop.computation.moire import generate_moire_pattern
from waytogocoop.computation.pdw import moire_locked_period
from waytogocoop.computation.spin_fluctuation import (
    DOPINGS_LCCO,
    SF_COHERENCE_MODELS,
    SF_DOPING_REGIMES,
    SF_GLUE_MOIRE_PERIOD,
    SF_INELASTIC_FORMS,
    SF_PAIRING_FORMS,
    SfConfig,
    SfGlueConfig,
    SfPairingMap,
    SfQuadratureConfig,
    bcs_route_relative_modulation,
    calibration_points,
    coherence_factor,
    coherence_length,
    conductivity_tensor,
    cvc_centres,
    cvc_epsilon_profile,
    cvc_partner_angle,
    cvc_velocity,
    doping_regime,
    doping_sweep,
    drude_hall_coefficient,
    drude_hall_coefficient_nominal,
    drude_resistivity_per_rate,
    equivalent_doping,
    fermi_surface,
    fermi_surface_at,
    field_derivative,
    field_grid,
    film_of_x,
    fit_resistivity_polynomial,
    fragmentation_tc,
    g_of_x,
    glue_gap_scale,
    glue_modulated_gap,
    hall_coefficient,
    hall_coefficient_chambers,
    hall_coefficient_curve,
    hall_ratio_to_case_a,
    high_field_slope,
    hotspot_angles,
    imprint_curve,
    inelastic_weight,
    lambda_from_tc,
    lambda_resistive,
    lambda_threshold,
    lambda_tr,
    linearized_glue_amplitude,
    local_imprint,
    log_sensitivity,
    luttinger_holes,
    magnetoresistance,
    moire_profile,
    pairing_fit_rms,
    quadrature_field_derivative,
    quadrature_resistivity,
    resistivity_fit,
    resistivity_tensor,
    scattering_rate,
    snap_doping,
    tc_ceiling,
    tc_from_lambda,
    tc_of_x,
    zero_field_resistivity,
    zone_boundary_crossing_deg,
)
from waytogocoop.computation.superconducting import cpdm_amplitude, gap_modulation

pytestmark = pytest.mark.speculative

DEFAULT = SfConfig()
CASE_A = replace(DEFAULT, inelastic="none", hotspots=False)  # Fig. 13a: Gamma_0 only
CASE_B = replace(DEFAULT, inelastic="none")  # 13b: + hot spots
CASE_C = replace(DEFAULT, inelastic="antinodal")  # 13c: + antinodal inelastic
KOHLER = replace(DEFAULT, inelastic="isotropic", hotspots=False)
POWER_LAW = SfPairingMap()
MCMILLAN = SfPairingMap(form="mcmillan")
I20, I33, I35 = 40, 66, 70  # 20, 33 and 35 T on the default 0.5 T grid
MR_TEMPS = (20.0, 30.0, 40.0, 50.0)
T_1K = np.arange(2.0, 201.0, 1.0)

# Fig. 4b2-b4 simulations at x = 0.159 (vector read), at 20 / 30 / 40 / 50 K.
FIG4_RHO0 = (19.99, 22.15, 24.42, 26.78)
FIG4_MR20 = (0.0195, 0.0211, 0.0234, 0.0260)
FIG4_MR33 = (0.0387, 0.0444, 0.0505, 0.0564)
# Fig. 5a/5b simulations at 40 K, 20 T: x -> (rho(0) micro-Ohm cm, MR), the 13 S1 dopings
# (vector read; 0.119, 0.134, 0.141, 0.149 and 0.156 raster read to +-0.3 / +-0.0004).
FIG5 = {
    0.116: (43.6, 0.0421), 0.119: (41.3, 0.0410), 0.120: (39.6, 0.0409),
    0.125: (35.2, 0.0368), 0.131: (31.9, 0.0334), 0.134: (30.6, 0.0316),
    0.138: (29.0, 0.0300), 0.141: (28.1, 0.0290), 0.146: (26.4, 0.0271),
    0.149: (26.3, 0.0264), 0.153: (26.0, 0.0259), 0.156: (25.2, 0.0251),
    0.159: (24.4, 0.0239),
}  # fmt: skip


def _rel(want: float, rel: float):
    """Relative tolerance only: pytest.approx keeps a 1e-12 absolute floor otherwise."""
    return pytest.approx(want, rel=rel, abs=0.0)


@cache
def _family(cfg: SfConfig):
    return magnetoresistance(cfg)


@cache
def _sweep():
    return doping_sweep(DEFAULT)


@cache
def _hall(cfg: SfConfig, cvc: bool) -> np.ndarray:
    return hall_coefficient_curve(cfg, T_1K, cvc)


@cache
def _pattern() -> np.ndarray:
    """The repo's default moire pattern: FeTe (square) under Sb2Te3 (hexagonal), 200^2."""
    pattern = generate_moire_pattern(3.82, 4.264, "hexagonal", "square", 0.0, 200, 100.0)
    return np.asarray(pattern["pattern"], dtype=float)


def _mr(cfg: SfConfig, temperature: float, field: float) -> float:
    rho = resistivity_tensor(cfg, temperature, [0.0, field])[:, 0]
    return float(rho[1] / rho[0] - 1.0)


class TestDopingTables:
    def test_g_table(self):
        assert len(DOPINGS_LCCO) == 19
        assert list(DOPINGS_LCCO) == sorted(DOPINGS_LCCO)
        g = [g_of_x(x) for x in DOPINGS_LCCO]
        assert g_of_x(0.159) == 1.0
        assert g[0] == 6.50 and g[-1] == 0.55
        assert np.all(np.diff(g) <= 0.0)

    def test_g_interpolates_linearly_and_is_exact_at_nodes(self):
        assert g_of_x(0.138) == 2.10
        assert g_of_x(0.141 + 5e-13) == 1.87
        assert g_of_x(0.14) == pytest.approx(2.10 + (1.87 - 2.10) * (0.002 / 0.003), rel=1e-12)
        assert g_of_x(0.16) == pytest.approx(1.00 + (0.85 - 1.00) * (0.001 / 0.006), rel=1e-12)

    @pytest.mark.parametrize("x", [0.115, 0.189, np.nan, np.inf])
    def test_g_outside_the_table_raises(self, x):
        with pytest.raises(ValueError):
            g_of_x(x)

    def test_tc_table(self):
        assert tc_of_x(0.05) == 25.0
        assert tc_of_x(0.11) == 25.0  # inferred plateau below 0.12
        assert tc_of_x(0.12) == 25.0
        assert tc_of_x(0.14) == 15.0
        assert tc_of_x(0.16) == 8.0
        assert tc_of_x(0.15) == pytest.approx(11.5, rel=1e-12)
        assert tc_of_x(0.18) == 0.0
        assert tc_of_x(0.185) == 0.0
        assert tc_of_x(0.25) == 0.0

    def test_snap_doping(self):
        assert snap_doping(0.15) == 0.149
        assert snap_doping(0.159) == 0.159
        assert snap_doping("0.116") == 0.116
        assert snap_doping(0.5) == 0.188
        assert snap_doping(0.0) == 0.116
        for garbage in (np.nan, np.inf, None, "x = 0.15", [0.15]):
            assert snap_doping(garbage) == 0.159

    @pytest.mark.parametrize(("lower", "upper"), [(0.184, 0.188), (0.141, 0.146)])
    def test_snap_doping_tie_goes_to_the_lower_x(self, lower, upper):
        mid = 0.5 * (lower + upper)
        assert abs(mid - lower) == abs(upper - mid)  # a bitwise tie
        assert snap_doping(mid) == lower

    def test_doping_regimes(self):
        assert doping_regime(0.138) == "fs_reconstruction"
        assert doping_regime(0.141) == "superconducting"
        assert doping_regime(0.174) == "superconducting"
        assert doping_regime(0.180) == "beyond_dome"
        assert {doping_regime(x) for x in DOPINGS_LCCO} == set(SF_DOPING_REGIMES)

    def test_film_of_x(self):
        assert film_of_x(0.116) == "S1"
        assert film_of_x(0.159) == "S1"
        assert film_of_x(0.165) == "S2"


class TestFermiSurface:
    def test_extrema(self):
        kf, vf, *_ = fermi_surface_at(np.array([0.0, np.pi / 4.0]))
        assert kf[0] == DEFAULT.kf0
        assert kf[1] == _rel(DEFAULT.kf0 * (1.0 + DEFAULT.kf_anisotropy), 1e-15)
        assert vf[0] == DEFAULT.vf0
        assert vf[1] == _rel(DEFAULT.vf0 * (1.0 + DEFAULT.vf_anisotropy), 1e-15)
        fs = fermi_surface()
        assert fs.kf.min() == DEFAULT.kf0  # antinode k_F < node k_F (Fig. 4a1)
        assert fs.kf.max() == _rel(DEFAULT.kf0 * 1.11, 1e-15)

    def test_velocity_is_normal_to_the_surface(self):
        fs = fermi_surface()
        dk = 2.0 * DEFAULT.kf0 * DEFAULT.kf_anisotropy * np.sin(4.0 * fs.phi)  # d kF / d phi
        tx = dk * np.cos(fs.phi) - fs.kf * np.sin(fs.phi)
        ty = dk * np.sin(fs.phi) + fs.kf * np.cos(fs.phi)
        cosine = (fs.vx * tx + fs.vy * ty) / (np.hypot(fs.vx, fs.vy) * np.hypot(tx, ty))
        assert np.max(np.abs(cosine)) < 1e-12

    def test_velocity_points_outward_and_weights_are_positive(self):
        fs = fermi_surface()
        assert np.all(fs.vx * np.cos(fs.phi) + fs.vy * np.sin(fs.phi) > 0.0)
        assert np.all(fs.weight > 0.0) and np.all(fs.omega_per_tesla > 0.0)
        assert fs.phi.shape == (1024,)

    def test_luttinger_count(self):
        holes = luttinger_holes()
        assert holes == _rel(0.8391564248900439, 1e-12)
        assert abs(holes - (1.0 - 0.159)) < 0.005  # 1 - x = 0.841 (p. 20)

    def test_zone_boundary_crossing(self):
        crossing = zone_boundary_crossing_deg()
        assert crossing == _rel(27.196569908994878, 1e-12)
        assert abs(crossing - DEFAULT.phi_hs_deg) < 0.5
        phi = math.radians(crossing)
        kf = fermi_surface_at(phi)[0]
        assert kf * (math.cos(phi) + math.sin(phi)) == _rel(math.pi / 4.01e-10, 1e-14)

    @pytest.mark.parametrize("kf0", [4.0e9, 8.0e9])
    def test_no_crossing_raises(self, kf0):
        with pytest.raises(ValueError):
            zone_boundary_crossing_deg(replace(DEFAULT, kf0=kf0))

    def test_hotspot_angles_order_and_symmetry(self):
        p = 27.5
        want = [p, 90 - p, p + 90, 180 - p, p + 180, 270 - p, p + 270, 360 - p]
        np.testing.assert_allclose(np.degrees(hotspot_angles()), want, rtol=1e-14)
        hs = hotspot_angles()
        # mirror about the zone diagonal (phi <-> pi/2 - phi) and four-fold rotation
        np.testing.assert_allclose(hs[0::2] + hs[1::2], np.pi / 2 + np.arange(4) * np.pi)
        np.testing.assert_allclose(np.diff(hs[0::2]), np.pi / 2)


class TestScatteringRate:
    def test_peaks_at_the_eight_hot_spots(self):
        for centre in hotspot_angles():
            phi = centre + np.linspace(-0.2, 0.2, 4001)
            gamma = scattering_rate(phi, 40.0)
            assert abs(phi[np.argmax(gamma)] - centre) < 1e-3

    def test_isotropic_without_hot_spots_is_flat(self):
        phi = fermi_surface().phi
        for t in (0.0, 20.0, 40.0):
            gamma = scattering_rate(phi, t, KOHLER)
            assert np.ptp(gamma) == 0.0
            assert gamma[0] == _rel(27.0 + 0.75 * t + 0.0112 * t * t, 1e-14)

    @pytest.mark.parametrize(
        ("form", "plain", "dos"),
        [
            ("afm_boundary", math.comb(12, 6) * 2 / 2**12, 0.4580029405789852),
            ("antinodal", 0.5, 0.4908459753825611),
            ("isotropic", 1.0, 1.0),
            ("none", 0.0, 0.0),
        ],
    )
    def test_form_phi_averages(self, form, plain, dos):
        # The forms no longer share an average: the alternatives are peak-normalised.
        fs = fermi_surface()
        w = inelastic_weight(fs.phi, replace(DEFAULT, inelastic=form))
        assert np.mean(w) == pytest.approx(plain, rel=1e-12, abs=1e-15)
        assert np.sum(w * fs.weight) / np.sum(fs.weight) == pytest.approx(dos, rel=1e-12, abs=1e-15)

    def test_antinodal_is_cos_squared_at_any_nu(self):
        phi = np.linspace(0.0, 2.0 * np.pi, 97)
        for nu in (2, 12, 24):
            w = inelastic_weight(phi, replace(DEFAULT, inelastic="antinodal", nu=nu))
            np.testing.assert_array_equal(w, np.cos(2.0 * phi) ** 2)
        assert np.all(inelastic_weight(phi, replace(DEFAULT, inelastic="isotropic")) == 1.0)

    def test_afm_boundary_weight_peaks_at_the_hot_spots(self):
        w = inelastic_weight(hotspot_angles())
        np.testing.assert_allclose(w, 1.0 + math.sin(math.radians(20.0)) ** 12, rtol=1e-12)

    def test_temperature_enters_only_through_the_inelastic_term(self):
        phi = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)
        cfg = replace(DEFAULT, doping=0.116)
        t = 37.0
        w = inelastic_weight(phi, cfg)
        delta = scattering_rate(phi, t, cfg) - scattering_rate(phi, 0.0, cfg)
        np.testing.assert_allclose(delta, (6.50 * 0.75 * t + 0.0112 * t * t) * w, rtol=1e-12)

    def test_g_scales_only_the_linear_term(self):
        phi = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)
        t = 30.0
        diff = scattering_rate(phi, t, replace(DEFAULT, doping=0.116)) - scattering_rate(phi, t)
        np.testing.assert_allclose(
            diff, (6.50 - 1.0) * 0.75 * t * inelastic_weight(phi), atol=1e-12
        )

    def test_hotspots_off_removes_gamma_hs(self):
        phi = hotspot_angles()
        on = scattering_rate(phi, 40.0)
        off = scattering_rate(phi, 40.0, replace(DEFAULT, hotspots=False))
        np.testing.assert_allclose(on - off, 31.4, rtol=1e-6)  # neighbours add < 1e-6
        np.testing.assert_array_equal(
            scattering_rate(phi, 40.0, replace(DEFAULT, hotspot_peak=0.0)), off
        )


class TestCvc:
    def test_centres_follow_the_crossing(self):
        crossing = math.radians(zone_boundary_crossing_deg())
        half = 0.5 * np.pi
        want = [crossing, half - crossing, crossing + half, np.pi - crossing, crossing + np.pi,
                3 * half - crossing, crossing + 3 * half, 2 * np.pi - crossing]  # fmt: skip
        np.testing.assert_allclose(cvc_centres(), want, rtol=1e-14)
        assert np.max(np.abs(cvc_centres() - hotspot_angles())) > 1e-3  # phi_HS = 27.5 deg

    def test_partner_of_each_centre_is_a_centre(self):
        centres = cvc_centres()
        for partner in cvc_partner_angle(centres):
            d = np.abs(np.mod(partner - centres + np.pi, 2.0 * np.pi) - np.pi)
            assert d.min() < 1e-12

    def test_epsilon_peaks_at_the_centres(self):
        np.testing.assert_allclose(
            cvc_epsilon_profile(cvc_centres()), DEFAULT.cvc_epsilon, rtol=1e-12
        )
        eps = cvc_epsilon_profile(cvc_centres(), replace(DEFAULT, cvc_epsilon=0.4))
        np.testing.assert_allclose(eps, 0.4, rtol=1e-12)

    def test_partner_of_zero_is_pi(self):
        # sgn(0) = 0, so the shift at phi = 0 is along -x only.
        assert cvc_partner_angle(0.0) == math.pi
        assert float(cvc_partner_angle(np.array([0.0]))[0]) == math.pi

    def test_dressed_current_at_the_centres_matches_fig_12a(self):
        # Fig. 12a plots v*_x in the inward (electron) convention: -J*_x here.
        centres = cvc_centres()
        eps = cvc_epsilon_profile(centres)
        _, _, _, vx, _ = fermi_surface_at(centres)
        _, _, _, pvx, _ = fermi_surface_at(cvc_partner_angle(centres))
        inward = -(vx + eps * pvx)[:4] / 1e5
        np.testing.assert_allclose(inward, [-3.4356, 3.4356, -3.4356, 3.4356], atol=0.15)

    def test_moving_phi_hs_leaves_epsilon_unchanged(self):
        phi = np.linspace(0.0, 2.0 * np.pi, 513)
        moved = replace(DEFAULT, phi_hs_deg=24.0)
        np.testing.assert_array_equal(cvc_epsilon_profile(phi, moved), cvc_epsilon_profile(phi))
        assert np.max(np.abs(hotspot_angles(moved) - hotspot_angles())) > 0.05

    def test_explicit_centres_change_nothing(self):
        phi = fermi_surface().phi
        np.testing.assert_array_equal(
            cvc_epsilon_profile(phi, DEFAULT, centres=cvc_centres(DEFAULT)),
            cvc_epsilon_profile(phi),
        )

    def test_bracket_without_crossing_raises(self):
        with pytest.raises(ValueError):
            cvc_centres(replace(DEFAULT, kf0=4.0e9))
        with pytest.raises(ValueError):
            hall_coefficient(replace(DEFAULT, kf0=8.0e9), 40.0, cvc=True)

    def test_bad_centres_raise(self):
        with pytest.raises(ValueError):
            cvc_epsilon_profile(0.1, DEFAULT, centres=cvc_centres()[:4])
        with pytest.raises(ValueError):
            cvc_epsilon_profile(0.1, DEFAULT, centres=np.full(8, np.nan))

    def test_cvc_velocity_dresses_the_bare_velocity(self):
        fs = fermi_surface()
        jx, jy = cvc_velocity()
        _, _, _, pvx, pvy = fermi_surface_at(cvc_partner_angle(fs.phi))
        eps = cvc_epsilon_profile(fs.phi)
        np.testing.assert_array_equal(jx, fs.vx + eps * pvx)
        np.testing.assert_array_equal(jy, fs.vy + eps * pvy)
        jx0, jy0 = cvc_velocity(replace(DEFAULT, cvc_epsilon=0.0))
        np.testing.assert_array_equal(jx0, fs.vx)
        np.testing.assert_array_equal(jy0, fs.vy)


class TestChambers:
    CIRCLE = replace(
        DEFAULT, kf_anisotropy=0.0, vf_anisotropy=0.0, inelastic="none", hotspots=False
    )

    def test_drude_circle(self):
        fs = fermi_surface(self.CIRCLE)
        sigma = conductivity_tensor(self.CIRCLE, 0.0, [0.0, 2.0, 20.0, 200.0])
        s0, tau = sigma[0, 0, 0], 1.0 / 27.0e12
        for k, b in enumerate((2.0, 20.0, 200.0), start=1):
            wt = fs.omega_per_tesla[0] * b * tau
            assert sigma[k, 0, 0] == _rel(s0 / (1.0 + wt * wt), 1e-10)
            assert abs(sigma[k, 1, 0]) == _rel(s0 * wt / (1.0 + wt * wt), 1e-10)

    def test_isotropic_circle_has_no_magnetoresistance(self):
        rho = resistivity_tensor(self.CIRCLE, 0.0, [0.0, 5.0, 35.0])[:, 0]
        assert np.max(np.abs(rho / rho[0] - 1.0)) <= 1e-10

    def test_onsager_symmetry(self):
        sigma = conductivity_tensor(DEFAULT, 40.0, [20.0, -20.0])
        assert sigma[0, 0, 0] == _rel(sigma[1, 0, 0], 1e-13)
        assert sigma[0, 0, 1] == _rel(-sigma[1, 0, 1], 1e-13)

    def test_zero_field_closed_form(self):
        rho = resistivity_tensor(DEFAULT, 40.0, [0.0])[0]
        assert rho[0] == _rel(float(zero_field_resistivity(DEFAULT, [40.0])[0]), 1e-14)
        assert abs(rho[1]) < 1e-12 * rho[0]  # sigma_xy(0) is an odd sum: zero up to round-off

    def test_no_spurious_linear_term(self):
        # sigma_xx is quadratic at low field: no fake H-linear MR from a first-order scheme.
        sigma = conductivity_tensor(DEFAULT, 40.0, [0.0, 1e-4, 0.1])
        coeff = [(sigma[k, 0, 0] / sigma[0, 0, 0] - 1.0) / b**2 for k, b in ((1, 1e-4), (2, 0.1))]
        assert abs(coeff[1] / coeff[0] - 1.0) < 1e-3
        assert coeff[0] < 0.0

    def test_vectorised_solve_equals_single_fields(self):
        fields = [0.0, 0.5, -3.0, 20.0, -35.0]
        batch = conductivity_tensor(DEFAULT, 40.0, fields)
        for k, b in enumerate(fields):
            np.testing.assert_array_equal(batch[k], conductivity_tensor(DEFAULT, 40.0, [b])[0])

    def test_family_rows_equal_single_temperature_solves(self):
        fam = _family(DEFAULT)
        rho = resistivity_tensor(DEFAULT, 30.0, fam.fields)
        np.testing.assert_array_equal(fam.rho_xx[1], rho[:, 0])
        np.testing.assert_array_equal(fam.rho_yx[1], rho[:, 1])

    def test_convergence_in_n_phi(self):
        coarse = _mr(DEFAULT, 40.0, 20.0)
        fine = _mr(replace(DEFAULT, n_phi=2048), 40.0, 20.0)
        assert abs(coarse / fine - 1.0) < 5e-4

    def test_field_grid_and_derivative(self):
        grid = field_grid()
        assert grid.size == 71 and grid[0] == 0.0 and grid[-1] == 35.0
        assert field_grid(35.3, 0.5)[-1] == 35.0  # snapped down
        rho = np.cos(np.linspace(0.0, 3.0, 17)) ** 3
        np.testing.assert_allclose(
            field_derivative(rho, 0.25), np.gradient(rho, 0.25, edge_order=2), rtol=1e-14
        )

    def test_magnetoresistance_layout(self):
        fam = _family(DEFAULT)
        assert fam.rho_xx.shape == fam.mr.shape == fam.drho_dh.shape == (4, 71)
        np.testing.assert_array_equal(fam.temperatures, MR_TEMPS)
        np.testing.assert_array_equal(fam.mr[:, 0], 0.0)
        np.testing.assert_allclose(fam.h_over_t[2], fam.fields / 40.0)
        assert np.all(fam.rho_yx[:, 1:] > 0.0)  # holes: R_H > 0
        empty = magnetoresistance(DEFAULT, [], field_max=5.0)
        assert empty.rho_xx.shape == (0, 11)
        with pytest.raises(ValueError):
            magnetoresistance(DEFAULT, MR_TEMPS, field_max=0.5)


class TestHall:
    def test_jones_zener_equals_low_field_chambers(self):
        for cfg, tol in ((CASE_A, 1e-6), (DEFAULT, 1e-4)):
            for t in (0.0, 40.0):
                jz = hall_coefficient(cfg, t, cvc=False)
                assert hall_coefficient_chambers(cfg, t, 0.1) == _rel(jz, tol)

    def test_isotropic_close_to_drude(self):
        assert hall_coefficient(CASE_A, 0.0, cvc=False) == _rel(drude_hall_coefficient(), 0.02)

    def test_nominal_drude(self):
        assert drude_hall_coefficient_nominal() == _rel(7.3871e-10, 1e-4)
        assert drude_hall_coefficient_nominal(replace(DEFAULT, doping=0.116)) == _rel(
            7.0278e-10, 1e-4
        )
        assert drude_hall_coefficient() == _rel(7.403312411077447e-10, 1e-12)

    def test_positive_without_cvc_for_every_form(self):
        for form in SF_INELASTIC_FORMS:
            for hotspots in (True, False):
                cfg = replace(DEFAULT, inelastic=form, hotspots=hotspots)
                rh = hall_coefficient_curve(cfg, [2.0, 20.0, 50.0, 100.0, 200.0], cvc=False)
                assert np.all(rh > 0.0), (form, hotspots)

    def test_afm_boundary_falls_strictly_without_cvc(self):
        assert np.all(np.diff(_hall(DEFAULT, False)) < 0.0)

    def test_cvc_flattens_the_temperature_dependence(self):
        assert hall_coefficient(DEFAULT, 100.0) < hall_coefficient(DEFAULT, 40.0)
        assert hall_coefficient(DEFAULT, 40.0) < hall_coefficient(DEFAULT, 30.0)
        window = T_1K <= 100.0
        cvc, bare = _hall(DEFAULT, True)[window], _hall(DEFAULT, False)[window]
        assert cvc.max() / cvc.min() < 1.10  # measured 1.0926
        assert bare.max() / bare.min() > 1.40  # measured 1.4316

    def test_antinodal_rises_strictly(self):
        assert np.all(np.diff(_hall(CASE_C, False)) > 0.0)
        assert np.all(np.diff(_hall(CASE_C, True)) > 0.0)

    def test_fig13_order(self):
        def order(t: float, cvc: bool) -> bool:
            a, b, c, d = (
                hall_coefficient(cfg, t, cvc) for cfg in (CASE_A, CASE_B, CASE_C, DEFAULT)
            )
            return c > a > b > d

        assert order(100.0, True)
        assert order(50.0, False) and order(100.0, False)

    def test_cvc_lowers_cases_a_b_and_d(self):
        assert hall_coefficient(CASE_A, 0.0, False) == _rel(7.340e-10, 1e-3)
        assert hall_coefficient(CASE_A, 0.0, True) == _rel(5.309e-10, 1e-3)
        assert hall_coefficient(CASE_B, 0.0, False) == _rel(7.278e-10, 1e-3)
        assert hall_coefficient(CASE_B, 0.0, True) == _rel(4.929e-10, 1e-3)
        window = T_1K <= 100.0
        assert np.all(_hall(DEFAULT, True)[window] < _hall(DEFAULT, False)[window])

    def test_d_over_a_with_cvc(self):
        for t in (2.0, 50.0, 100.0):
            assert 0.85 <= hall_ratio_to_case_a(DEFAULT, t) <= 1.0

    def test_ratio_denominator_is_temperature_independent(self):
        for t in (0.0, 40.0, 200.0):
            assert hall_coefficient(CASE_A, t, cvc=True) == _rel(5.309220240734655e-10, 1e-12)
        assert hall_ratio_to_case_a(DEFAULT, 40.0) == _rel(0.9592004336484117, 1e-12)
        assert hall_ratio_to_case_a(DEFAULT, 40.0) == _rel(
            hall_coefficient(DEFAULT, 40.0) / hall_coefficient(CASE_A, 0.0), 1e-15
        )

    def test_chambers_needs_a_field(self):
        with pytest.raises(ValueError):
            hall_coefficient_chambers(DEFAULT, 40.0, 0.0)


class TestPaperReproduction:
    def test_rho0_against_fig_4b2(self):
        np.testing.assert_allclose(_family(DEFAULT).rho_xx[:, 0], FIG4_RHO0, rtol=0.01)

    def test_mr_against_fig_4b3(self):
        fam = _family(DEFAULT)
        np.testing.assert_allclose(fam.mr[:, I20], FIG4_MR20, rtol=0.06)
        np.testing.assert_allclose(fam.mr[:, I33], FIG4_MR33, rtol=0.06)

    def test_anti_kohler(self):
        fam = _family(DEFAULT)
        assert np.all(np.diff(fam.mr[:, I20]) > 0.0)
        assert np.all(np.diff(fam.mr[:, I35]) > 0.0)

    def test_isotropic_is_kohler_like(self):
        assert np.all(np.diff(_family(KOHLER).mr[:, I20]) < 0.0)

    def test_kohler_collapse_for_uniform_gamma(self):
        # Gamma uniform on the FS: MR is a function of B / rho0 alone (exact).
        mr = []
        for t in (20.0, 50.0):
            rho0 = float(zero_field_resistivity(KOHLER, [t])[0])
            mr.append(_mr(KOHLER, t, rho0))  # B = 1 T per micro-Ohm cm
        assert mr[1] == _rel(mr[0], 1e-10)

    def test_fig_9c_inverts_the_temperature_dependence(self):
        fam = _family(replace(DEFAULT, inelastic="isotropic", hotspot_peak=105.0))
        assert np.all(np.diff(fam.mr[:, I35]) < 0.0)
        assert fam.rho_xx[3, 0] / fam.rho_xx[0, 0] > 1.8  # rho too steep (measured 1.917)

    def test_fig_9e_saturates_more(self):
        def saturation(cfg: SfConfig) -> float:
            slope = _family(cfg).drho_dh[0]  # 20 K
            return slope[-1] / slope.max()

        assert saturation(CASE_C) < 0.75  # measured 0.646
        assert saturation(CASE_C) < saturation(DEFAULT)  # paper form 0.890

    def test_fig_9b_hot_spots_carry_the_low_t_mr(self):
        bare = _family(replace(DEFAULT, hotspots=False))
        paper = _family(DEFAULT)
        assert bare.mr[0, I35] < 0.6 * paper.mr[0, I35]
        fan_bare = bare.drho_dh[3, 5] / bare.drho_dh[0, 5]  # initial slopes at 2.5 T
        fan_paper = paper.drho_dh[3, 5] / paper.drho_dh[0, 5]
        assert fan_bare > 2.5
        assert fan_paper < 2.0

    def test_doping_series_against_fig_5(self):
        sweep = _sweep()
        for i, x in enumerate(sweep.dopings):
            if x in FIG5:
                rho, mr = FIG5[x]
                assert sweep.rho0[i] == _rel(rho, 0.05), x
                assert sweep.mr[i] == _rel(mr, 0.13), x

    def test_s1_drops(self):
        sweep = _sweep()
        assert 1.5 <= sweep.rho0[0] / sweep.rho0[12] <= 2.0
        assert 1.5 <= sweep.mr[0] / sweep.mr[12] <= 2.0
        assert np.all(np.diff(sweep.mr[:13]) < 0.0)

    def test_no_h_over_t_collapse(self):
        fam = _family(DEFAULT)
        # mu0 H / T = 0.5 lands on the grid at 10, 15, 20 and 25 T.
        slopes = [fam.drho_dh[k, int(round(0.5 * t / 0.5))] for k, t in enumerate(MR_TEMPS)]
        assert max(slopes) / min(slopes) > 2.0

    def test_table2_height_breaks_anti_kohler(self):
        fam = _family(replace(DEFAULT, hotspot_peak=config.HOTSPOT_PEAK_TABLE2))
        assert fam.mr[0, I20] > fam.mr[1, I20]


class TestCoupling:
    def test_lambda_tr_docstring_names_the_coupling(self):
        # Naming rule: the full phrase lives in lambda_tr's docstring.
        phrase = (
            "lambda_tr, the Fermi-surface-averaged T-linear coupling "
            "(Allen's form applied to the microscopic rate)"
        )
        assert phrase in " ".join(lambda_tr.__doc__.split())

    def test_lambda_tr_against_an_independent_quadrature(self):
        n = 8192
        phi = 2.0 * np.pi * np.arange(n) / n
        s2 = np.sin(2.0 * phi) ** 2
        kf = 5.424e9 * (1.0 + 0.11 * s2)
        vf = 4.946e5 * (1.0 + 0.03 * s2)
        tilt = np.arctan(2.0 * 0.11 * np.sin(4.0 * phi) / (1.0 + 0.11 * s2))
        dos = kf / (vf * np.cos(tilt))
        off = np.pi / 4.0 - np.radians(27.5)
        w = np.sin(2.0 * (phi + off)) ** 12 + np.sin(2.0 * (phi - off)) ** 12
        want = (
            1.054571817e-34 * 0.75e12 * np.sum(w * dos) / np.sum(dos) / (2.0 * np.pi * 1.380649e-23)
        )
        assert lambda_tr() == _rel(want, 1e-12)

    def test_coefficient_and_forms(self):
        assert lambda_tr() == _rel(0.417582739923252, 1e-12)
        forms = {"afm_boundary": 0.41758273992325223, "antinodal": 0.4475272735616924,
                 "isotropic": 0.9117468534052734}  # fmt: skip
        for form, want in forms.items():
            assert lambda_tr(replace(DEFAULT, inelastic=form)) == _rel(want, 1e-12)
        assert lambda_tr(replace(DEFAULT, inelastic="none")) == 0.0

    def test_lambda_tr_strictly_increasing_in_g(self):
        x = np.linspace(0.116, 0.188, 145)
        lam = np.array([lambda_tr(replace(DEFAULT, doping=float(v))) for v in x])
        g = np.array([g_of_x(float(v)) for v in x])
        np.testing.assert_allclose(lam / g, lambda_tr(), rtol=1e-12)
        dg, dlam = np.diff(g), np.diff(lam)
        assert np.all(dlam[dg < 0.0] < 0.0)  # g falls with x, and lambda_tr with it
        assert np.all(dlam[dg == 0.0] == 0.0)

    def test_drude_resistivity_per_rate(self):
        rho_d = drude_resistivity_per_rate()
        assert rho_d == _rel(5.5005672382590704e-21, 1e-12)
        # rho = rho_D Gamma exactly for a uniform Gamma.
        gamma = (27.0 + 0.75 * 40.0 + 0.0112 * 1600.0) * 1e12
        rho = float(zero_field_resistivity(KOHLER, [40.0])[0]) / 1e8
        assert rho == _rel(rho_d * gamma, 1e-13)

    def test_lambda_rho_values(self):
        want = (0.4352438971760371, 1.491881542660157, 0.2946327374251149)
        for x, value in zip((0.159, 0.116, 0.188), want, strict=True):
            assert lambda_resistive(replace(DEFAULT, doping=x)) == _rel(value, 1e-9)

    @pytest.mark.parametrize("alpha2", [0.0, 0.0112])
    def test_lambda_rho_equals_lambda_tr_for_uniform_gamma(self, alpha2):
        cfg = replace(KOHLER, alpha2=alpha2)
        assert lambda_resistive(cfg) == _rel(lambda_tr(cfg), 1e-9)

    def test_doping_sweep(self):
        sweep = _sweep()
        assert sweep.dopings.shape == (19,) and sweep.temperature == 40.0 and sweep.field == 20.0
        np.testing.assert_array_equal(sweep.tc, [tc_of_x(x) for x in DOPINGS_LCCO])
        assert np.all(np.diff(sweep.lambda_rho / sweep.lambda_tr) >= 0.0)
        assert sweep.lambda_tr[0] / sweep.lambda_tr[-1] == pytest.approx(11.82, abs=0.01)
        assert sweep.lambda_rho[0] / sweep.lambda_rho[-1] == pytest.approx(5.06, abs=0.01)

    def test_doping_sweep_matches_single_doping_paths(self):
        sweep = _sweep()
        for i in (0, 12, 18):
            cfg = replace(DEFAULT, doping=DOPINGS_LCCO[i])
            rho = resistivity_tensor(cfg, 40.0, [0.0, 20.0])[:, 0]
            assert sweep.rho0[i] == rho[0]
            assert sweep.mr[i] == rho[1] / rho[0] - 1.0
            assert sweep.lambda_tr[i] == lambda_tr(cfg)
            assert sweep.lambda_rho[i] == lambda_resistive(cfg)
        zero = doping_sweep(DEFAULT, 40.0, 0.0)
        np.testing.assert_array_equal(zero.mr, 0.0)

    def test_resistivity_fit(self):
        fit = resistivity_fit(DEFAULT)
        assert (fit.t_min, fit.t_max) == (15.0, 60.0)
        assert fit.rho0 == _rel(15.80541402181645, 1e-9)
        assert fit.a1 == _rel(0.19693692765571835, 1e-9)
        assert fit.a2 == _rel(0.0005046269609317277, 1e-9)

    def test_polynomial_fit_is_exact_for_a_quadratic(self):
        t = np.arange(10.0, 31.0)
        fit = fit_resistivity_polynomial(t, 3.0 - 0.5 * t + 0.02 * t * t)
        assert (fit.rho0, fit.a1, fit.a2) == pytest.approx((3.0, -0.5, 0.02), rel=1e-9)
        for bad in (([1.0, 1.0, 2.0], [1.0, 2.0, 3.0]), ([1.0, 2.0, 3.0], [1.0, 2.0])):
            with pytest.raises(ValueError):
                fit_resistivity_polynomial(*bad)


class TestPairingMap:
    def test_power_law_passes_through_the_labels(self):
        points = calibration_points()
        assert [(x, tc) for x, tc, _ in points] == list(config.TC_LABELS_FIG5C)
        for _x, tc, lam in points[:3]:
            assert tc_from_lambda(lam) == pytest.approx(tc, rel=1e-9)
        assert tc_from_lambda(points[3][2]) == 0.0  # x = 0.18: lambda < lambda_c
        assert points[0][2] == POWER_LAW.lambda_ref

    def test_mcmillan_best_fit_fails_and_is_a_local_minimum(self):
        rms = pairing_fit_rms(MCMILLAN)
        assert rms == pytest.approx(2.907, abs=0.001)
        assert pairing_fit_rms(replace(MCMILLAN, omega_k=MCMILLAN.omega_k + 0.5)) > rms
        assert pairing_fit_rms(replace(MCMILLAN, omega_k=MCMILLAN.omega_k - 0.5)) > rms
        assert pairing_fit_rms(replace(MCMILLAN, mu_star=0.01)) > rms
        assert pairing_fit_rms(POWER_LAW) < 1e-9

    @pytest.mark.parametrize(
        "pairing", [POWER_LAW, MCMILLAN, SfPairingMap(form="mcmillan", mu_star=0.1)]
    )
    def test_round_trip(self, pairing):
        for tc in (0.5, 2.0, 8.0, 13.5, 25.0):
            lam = lambda_from_tc(tc, pairing)
            assert tc_from_lambda(lam, pairing) == pytest.approx(tc, rel=1e-12)
            assert lambda_from_tc(tc_from_lambda(lam, pairing), pairing) == pytest.approx(
                lam, rel=1e-12
            )

    def test_zero_at_and_below_the_threshold(self):
        assert lambda_threshold() == POWER_LAW.lambda_c
        assert tc_from_lambda(POWER_LAW.lambda_c) == 0.0
        np.testing.assert_array_equal(tc_from_lambda([0.0, 0.2, 0.3]), 0.0)
        strong = SfPairingMap(form="mcmillan", mu_star=0.1)
        threshold = lambda_threshold(strong)
        assert threshold == pytest.approx(0.1 / (1.0 - 0.062), rel=1e-15)
        assert tc_from_lambda(threshold, strong) == 0.0
        assert tc_from_lambda(0.5 * threshold, strong) == 0.0
        assert tc_from_lambda(0.5, strong) > 0.0  # (just above, exp(-1.04 (1 + l) / D) underflows)
        assert lambda_threshold(MCMILLAN) == 0.0

    @pytest.mark.parametrize("pairing", [POWER_LAW, MCMILLAN])
    def test_sensitivity_matches_a_finite_difference(self, pairing):
        for lam in (0.5, 1.0, 2.0):
            h = 1e-6 * lam
            fd = (
                math.log(tc_from_lambda(lam + h, pairing))
                - math.log(tc_from_lambda(lam - h, pairing))
            ) / (math.log(lam + h) - math.log(lam - h))
            assert fd == pytest.approx(log_sensitivity(lam, pairing), rel=1e-6)

    def test_ceiling(self):
        assert tc_ceiling() == math.inf
        assert tc_ceiling(MCMILLAN) == pytest.approx(43.39985, abs=1e-5)
        for bad in (43.4, 50.0, 0.0, -1.0):
            with pytest.raises(ValueError):
                lambda_from_tc(bad, MCMILLAN)
        with pytest.raises(ValueError):
            lambda_from_tc(0.0)

    def test_scalar_and_array_inputs(self):
        assert isinstance(tc_from_lambda(1.0), float)
        assert isinstance(log_sensitivity(1.0), float)
        assert tc_from_lambda(np.array([1.0, 2.0])).shape == (2,)

    def test_equivalent_doping(self):
        assert equivalent_doping(lambda_from_tc(13.5)) == _rel(0.1435584333754864, 1e-12)
        assert equivalent_doping(POWER_LAW.lambda_c) == _rel(0.17052577242235883, 1e-12)
        assert equivalent_doping(lambda_tr(replace(DEFAULT, doping=0.125))) == pytest.approx(
            0.125, rel=1e-12
        )
        for outside in (3.0, 0.2, np.nan, "bogus"):
            assert equivalent_doping(outside) is None


class TestGlueModulation:
    @pytest.mark.parametrize("coherence", SF_COHERENCE_MODELS)
    def test_uniform_pattern_gives_the_gap_scale(self, coherence):
        for tc in (2.0, 13.5):
            glue = SfGlueConfig(tc_target=tc, coherence=coherence)
            res = glue_modulated_gap(np.full((8, 8), 0.5), glue)
            np.testing.assert_allclose(res.gap, glue_gap_scale(glue), rtol=1e-12)

    def test_zero_eta_is_uniform(self):
        res = glue_modulated_gap(_pattern(), SfGlueConfig(eta=0.0))
        assert np.all(res.gap == res.gap.flat[0])
        assert res.gap.flat[0] == pytest.approx(config.DELTA_AVG, rel=1e-12)
        assert res.relative_modulation == 0.0

    @pytest.mark.parametrize("coherence", SF_COHERENCE_MODELS)
    @pytest.mark.parametrize("tc", [13.5, 5.0])
    def test_small_eta_identity(self, coherence, tc):
        glue = SfGlueConfig(eta=1e-4, tc_target=tc, coherence=coherence)
        res = glue_modulated_gap(_pattern(), glue)
        ref = gap_modulation(_pattern(), glue_gap_scale(glue), linearized_glue_amplitude(glue))
        np.testing.assert_allclose(res.gap, ref, rtol=1e-6)

    def test_coherence_factor_is_cpdm_amplitude_below_the_clamp(self):
        for tc in (2.0, 5.0, 13.5):
            glue = SfGlueConfig(tc_target=tc)
            xi = coherence_length(glue)
            assert xi / SF_GLUE_MOIRE_PERIOD <= 3.68
            assert coherence_factor(glue) == _rel(cpdm_amplitude(SF_GLUE_MOIRE_PERIOD, xi), 1e-15)

    def test_coherence_factor_is_not_clamped(self):
        glue = SfGlueConfig(tc_target=0.05)
        xi = coherence_length(glue)
        assert xi / SF_GLUE_MOIRE_PERIOD == pytest.approx(147.2, abs=0.01)
        assert coherence_factor(glue) == math.exp(-(xi / SF_GLUE_MOIRE_PERIOD))
        assert coherence_factor(glue) == 1.1845830137647655e-64
        assert cpdm_amplitude(SF_GLUE_MOIRE_PERIOD, xi) == pytest.approx(3.72e-44, rel=1e-3)

    def test_coherence_lengths(self):
        for tc, clean, dirty in ((13.5, 20.0, 20.0), (5.0, 54.0, 32.86335345030997),
                                 (2.0, 135.0, 51.96152422706632)):  # fmt: skip
            assert coherence_length(SfGlueConfig(tc_target=tc)) == _rel(clean, 1e-14)
            assert coherence_length(SfGlueConfig(tc_target=tc, coherence="dirty")) == _rel(
                dirty, 1e-14
            )
            assert coherence_length(SfGlueConfig(tc_target=tc, coherence="local")) == _rel(
                clean, 1e-14
            )
            assert coherence_factor(SfGlueConfig(tc_target=tc, coherence="local")) == 1.0

    def test_gap_scale_keeps_the_film_ratio(self):
        # 2 Delta_bar / (kB Tc*) with config.KB_EV_K (KB_J_K / e gives 1.7e-11 less).
        for tc in (2.0, 5.0, 13.5):
            glue = SfGlueConfig(tc_target=tc)
            ratio = 2.0 * glue_gap_scale(glue) * 1e-3 / (config.KB_EV_K * glue.tc_target)
            assert ratio == _rel(5.312290517954645, 1e-12)
        assert glue_gap_scale() == config.DELTA_AVG

    def test_default_moire_period_is_the_shared_period(self):
        assert SfGlueConfig().moire_period == moire_locked_period(3.82, 4.264, 0.0)
        assert SF_GLUE_MOIRE_PERIOD == 36.68576576576573

    @pytest.mark.parametrize("coherence", SF_COHERENCE_MODELS)
    def test_dressed_relative_is_f_times_local(self, coherence):
        for tc in (2.0, 5.0, 13.5):
            glue = SfGlueConfig(tc_target=tc, coherence=coherence)
            res = glue_modulated_gap(_pattern(), glue)
            f = coherence_factor(glue)
            assert res.coherence_factor == f
            assert res.relative_modulation == _rel(f * res.local_relative_modulation, 1e-12)
            exact, _ = local_imprint(glue.eta, tc)
            assert res.relative_modulation == _rel(f * exact, 1e-12)  # pattern spans [0, 1]

    @pytest.mark.parametrize(
        ("tc", "coherence", "relative", "local_relative", "local_normal", "gap_min", "gap_max"),
        [
            (13.5, "clean", 0.04081885749393932, 0.07040855267539989, 0.0,
             2.9569962300271597, 3.2092567693397047),
            (13.5, "dirty", 0.04081885749393932, 0.07040855267539989, 0.0,
             2.9569962300271597, 3.2092567693397047),
            (13.5, "local", 0.07040855267539989, 0.07040855267539989, 0.0,
             2.860581453791304, 3.295706309325275),
            (5.0, "clean", 0.15306744335253814, 0.6670327744083371, 0.0041,
             0.8818228883664684, 1.2321772587067223),
            (5.0, "dirty", 0.27233488876089784, 0.6670327744083371, 0.0041,
             0.6771928153962904, 1.300537116337901),
            (5.0, "local", 0.6670327744083371, 0.6670327744083371, 0.0041,
             0.0, 1.5267639058679714),
            (2.0, "clean", 0.032619065945499626, 1.293100808474302, 0.3062,
             0.4462301216137267, 0.4760946886571619),
            (2.0, "dirty", 0.31368949890574194, 1.293100808474302, 0.3062,
             0.34672682120871845, 0.6339269846513088),
            (2.0, "local", 1.293100808474302, 1.293100808474302, 0.3062,
             0.0, 1.1839056290920276),
        ],
    )  # fmt: skip
    def test_default_pattern_table(
        self, tc, coherence, relative, local_relative, local_normal, gap_min, gap_max
    ):
        # SF-amendments §4.3 "On the default pattern" (Python-only, pinned at 1e-9);
        # the pattern spans [0, 1], so the gap extremes are the P10 values.
        res = glue_modulated_gap(_pattern(), SfGlueConfig(tc_target=tc, coherence=coherence))
        assert res.relative_modulation == _rel(relative, 1e-9)
        assert res.local_relative_modulation == _rel(local_relative, 1e-9)
        assert res.local_normal_fraction == local_normal  # a pixel count over 200^2
        assert res.normal_fraction == (res.local_normal_fraction if coherence == "local" else 0.0)
        for value, want in ((res.gap.min(), gap_min), (res.gap.max(), gap_max)):
            assert value == 0.0 if want == 0.0 else value == _rel(want, 1e-9)
        assert res.gap.shape == res.tc_map.shape == res.lambda_map.shape == (200, 200)
        assert res.moire_period == SF_GLUE_MOIRE_PERIOD

    def test_normal_fractions(self):
        for tc in (2.0, 5.0, 13.5):
            for coherence in ("clean", "dirty"):
                res = glue_modulated_gap(
                    _pattern(), SfGlueConfig(tc_target=tc, coherence=coherence)
                )
                assert res.normal_fraction == 0.0
                assert res.gap.min() > 0.0
        local = SfGlueConfig(coherence="local")
        assert (
            glue_modulated_gap(_pattern(), replace(local, tc_target=5.1)).local_normal_fraction
            > 0.0
        )
        assert (
            glue_modulated_gap(_pattern(), replace(local, tc_target=5.2)).local_normal_fraction
            == 0.0
        )
        assert fragmentation_tc() == pytest.approx(5.177, abs=1e-3)

    @pytest.mark.parametrize("coherence", SF_COHERENCE_MODELS)
    def test_monotone_in_m(self, coherence):
        m = np.linspace(0.0, 1.0, 101)
        for tc in (2.0, 13.5):
            gap = glue_modulated_gap(m, SfGlueConfig(tc_target=tc, coherence=coherence)).gap
            assert np.all(np.diff(gap) >= 0.0)
        np.testing.assert_allclose(moire_profile(m), -np.cos(np.pi * m))

    def test_imprint_curve(self):
        tc = np.linspace(0.5, 25.0, 80)
        for coherence in SF_COHERENCE_MODELS:
            glue = SfGlueConfig(coherence=coherence)
            dressed, local, linear = imprint_curve(glue, tc)
            factor = [coherence_factor(replace(glue, tc_target=t)) for t in tc]
            np.testing.assert_allclose(dressed, np.array(factor) * local, rtol=1e-15)
            assert np.all(linear > 0.0)
        reference = bcs_route_relative_modulation()
        assert np.all(imprint_curve(SfGlueConfig(), tc)[0] < reference)
        assert imprint_curve(SfGlueConfig(coherence="dirty"), [2.0])[0][0] > reference
        assert imprint_curve(SfGlueConfig(coherence="local"), [5.0])[0][0] > reference

    def test_clean_maximum_sits_on_the_fragmentation_onset(self):
        tc = np.linspace(4.0, 7.0, 3001)
        dressed = imprint_curve(SfGlueConfig(), tc)[0]
        assert abs(tc[np.argmax(dressed)] - fragmentation_tc()) <= 2e-3

    def test_mcmillan_never_fragments(self):
        glue = SfGlueConfig(pairing=MCMILLAN)
        assert fragmentation_tc(glue) == 0.0
        assert fragmentation_tc(SfGlueConfig(eta=0.0)) == 0.0
        dressed = imprint_curve(glue, np.linspace(0.5, 25.0, 80))[0]
        assert np.all(dressed < bcs_route_relative_modulation())

    def test_amplitude_route_reference(self):
        assert bcs_route_relative_modulation() == _rel(0.51 / 3.09, 1e-15)
        assert bcs_route_relative_modulation() == _rel(0.1650485436893204, 1e-15)

    def test_longer_moire_period_lifts_the_clean_curve_above_the_reference(self):
        # The 0.165 comparison is conditional on L_m: at twice the shared period the
        # clean dressed curve crosses it.
        glue = SfGlueConfig(moire_period=2.0 * SF_GLUE_MOIRE_PERIOD)
        dressed = imprint_curve(glue, np.linspace(0.5, 25.0, 80))[0]
        assert int(np.sum(dressed > bcs_route_relative_modulation())) == 15
        assert dressed.max() == pytest.approx(0.3221, abs=1e-4)

    def test_local_imprint(self):
        exact, linear = local_imprint(0.1, 13.5)
        assert exact == _rel(0.0704085526753999, 1e-12)
        assert linear == _rel(0.07004041697562871, 1e-12)
        assert local_imprint(0.0, 13.5) == (0.0, 0.0)


class TestQuadratureContrast:
    def test_collapses_against_h_over_t(self):
        temps = np.array([10.0, 20.0, 40.0, 80.0])
        slopes = quadrature_field_derivative(temps, 0.5 * temps)
        np.testing.assert_allclose(slopes, slopes[0], rtol=1e-12)

    def test_saturates_at_gamma_mu_b(self):
        q = SfQuadratureConfig()
        assert quadrature_field_derivative(2.0, 1e6) == pytest.approx(q.gamma_mub, rel=1e-9)
        assert quadrature_field_derivative(40.0, 0.0) == 0.0
        assert quadrature_field_derivative(0.0, 0.0) == 0.0
        assert quadrature_field_derivative(0.0, 5.0) == pytest.approx(q.gamma_mub, rel=1e-15)

    def test_resistivity_form(self):
        q = SfQuadratureConfig()
        want = (
            q.rho0 + q.a_g * 40.0 + q.b * 1600.0 + math.hypot(q.alpha_kb * 40.0, q.gamma_mub * 20.0)
        )
        assert quadrature_resistivity(40.0, 20.0) == _rel(want, 1e-14)
        assert quadrature_resistivity(np.array([40.0, 50.0]), 20.0).shape == (2,)


class TestValidation:
    @pytest.mark.parametrize(
        "overrides",
        [
            {"doping": 0.115}, {"doping": 0.19}, {"doping": np.nan},
            {"inelastic": "bogus"}, {"hotspots": "yes"}, {"gamma_imp": "fast"},
            {"gamma_imp": 0.0}, {"gamma_imp": np.nan}, {"gamma_imp": np.inf},
            {"alpha1": -0.1}, {"alpha1": np.nan}, {"alpha2": -1e-4}, {"hotspot_peak": -1.0},
            {"hotspot_width": 0.0}, {"hotspot_width": np.nan}, {"cvc_sigma": 0.0},
            {"kf0": 0.0}, {"vf0": -1.0}, {"layer_spacing": 0.0}, {"layer_spacing": np.nan},
            {"nu": 13}, {"nu": 0}, {"nu": 26}, {"nu": 12.0}, {"nu": True},
            {"phi_hs_deg": 0.0}, {"phi_hs_deg": 45.0}, {"phi_hs_deg": np.nan},
            {"kf_anisotropy": 1.0}, {"kf_anisotropy": -0.1}, {"vf_anisotropy": np.nan},
            {"cvc_epsilon": 1.01}, {"cvc_epsilon": -0.01}, {"cvc_epsilon": np.nan},
            {"n_phi": 1000}, {"n_phi": 32}, {"n_phi": 16384}, {"n_phi": 1024.0},
            # float() parses these, the model's arithmetic does not
            {"doping": "0.159"}, {"gamma_imp": "27"}, {"kf0": b"5.424e9"},
            {"alpha1": Decimal("0.75")}, {"gamma_imp": 10**400},
        ],
    )  # fmt: skip
    def test_invalid_config_raises(self, overrides):
        cfg = SfConfig(**overrides)
        with pytest.raises(ValueError):
            fermi_surface(cfg)
        with pytest.raises(ValueError):
            resistivity_tensor(cfg, 40.0, [0.0, 20.0])

    @pytest.mark.parametrize("temperature", [-1.0, np.nan, np.inf, "warm", [20.0, 30.0]])
    def test_invalid_temperature_raises(self, temperature):
        with pytest.raises(ValueError):
            scattering_rate(0.1, temperature)
        with pytest.raises(ValueError):
            magnetoresistance(DEFAULT, [20.0, temperature])
        with pytest.raises(ValueError):
            hall_coefficient(DEFAULT, temperature)

    @pytest.mark.parametrize("fields", [[0.0, np.nan], [np.inf], [[1.0, 2.0]], ["strong"]])
    def test_invalid_fields_raise(self, fields):
        with pytest.raises(ValueError):
            conductivity_tensor(DEFAULT, 40.0, fields)

    @pytest.mark.parametrize(
        ("field_max", "step"), [(0.0, 0.5), (35.0, 0.0), (np.nan, 0.5), (35.0, -1.0)]
    )
    def test_invalid_field_grid_raises(self, field_max, step):
        with pytest.raises(ValueError):
            field_grid(field_max, step)

    def test_invalid_angles_and_arrays_raise(self):
        for fn in (scattering_rate, inelastic_weight, cvc_partner_angle, cvc_epsilon_profile):
            with pytest.raises(ValueError):
                fn(np.array([0.1, np.nan]), *((40.0,) if fn is scattering_rate else ()))
        with pytest.raises(ValueError):
            fermi_surface_at(np.inf)
        with pytest.raises(ValueError):
            scattering_rate("north", 40.0)
        with pytest.raises(ValueError):
            field_derivative([1.0, 2.0], 0.5)
        with pytest.raises(ValueError):
            resistivity_fit(DEFAULT, (60.0, 15.0))
        with pytest.raises(ValueError):
            high_field_slope(DEFAULT, 40.0, step=0.0)

    @pytest.mark.parametrize(
        "overrides",
        [
            {"form": "bcs"}, {"lambda_c": -0.1}, {"lambda_c": np.nan},
            {"lambda_ref": 0.3}, {"exponent": 0.0}, {"tc_ref": -1.0},
            {"omega_k": 0.0}, {"omega_k": np.nan}, {"mu_star": -0.1}, {"mu_star": 1.0 / 0.62},
            {"lambda_c": "0.3"}, {"tc_ref": "25"},
        ],
    )  # fmt: skip
    def test_invalid_pairing_raises(self, overrides):
        pairing = SfPairingMap(**overrides)
        with pytest.raises(ValueError):
            tc_from_lambda(1.0, pairing)
        with pytest.raises(ValueError):
            glue_modulated_gap(_pattern(), SfGlueConfig(pairing=pairing))

    @pytest.mark.parametrize(
        "overrides",
        [
            {"coherence": "warm"},
            {"moire_period": 0.0}, {"moire_period": -1.0},
            {"moire_period": np.inf}, {"moire_period": np.nan},
            {"delta_avg": 0.0}, {"delta_avg": np.nan},
            {"eta": 1.0}, {"eta": -0.1}, {"eta": np.nan},
            {"tc_target": 0.0}, {"tc_target": np.nan}, {"tc_target": -2.0},
            {"tc_target": 50.0, "pairing": SfPairingMap(form="mcmillan")},
            {"tc_target": "13.5"}, {"eta": "0.1"}, {"delta_avg": "3.09"}, {"moire_period": "36.7"},
            # Tc(lambda_bar) = 0: the inverse rounds onto lambda_c, or x^p underflows
            {"tc_target": 1e-6}, {"tc_target": 5.0, "pairing": SfPairingMap(exponent=0.001)},
        ],
    )  # fmt: skip
    def test_invalid_glue_raises(self, overrides):
        glue = SfGlueConfig(**overrides)
        for fn in (
            coherence_length,
            coherence_factor,
            glue_gap_scale,
            linearized_glue_amplitude,
            fragmentation_tc,
        ):
            with pytest.raises(ValueError):
                fn(glue)
        with pytest.raises(ValueError):
            glue_modulated_gap(_pattern(), glue)
        with pytest.raises(ValueError):
            imprint_curve(glue, [2.0])

    @pytest.mark.parametrize("targets", [[0.0], [-1.0, 2.0], [np.nan], [[2.0, 3.0]], [13.5, 1e-6]])
    def test_invalid_imprint_targets_raise(self, targets):
        with pytest.raises(ValueError):
            imprint_curve(SfGlueConfig(), targets)
        with pytest.raises(ValueError):
            imprint_curve(SfGlueConfig(pairing=MCMILLAN), [50.0])

    def test_invalid_pattern_and_imprint_inputs_raise(self):
        with pytest.raises(ValueError):
            glue_modulated_gap(np.array([[0.0, np.nan]]))
        with pytest.raises(ValueError):
            glue_modulated_gap(np.zeros((0, 3)))
        with pytest.raises(ValueError):
            local_imprint(1.0, 13.5)
        with pytest.raises(ValueError):
            local_imprint(0.1, 0.0)
        with pytest.raises(ValueError):
            local_imprint("0.1", 13.5)
        with pytest.raises(ValueError, match="resolvable"):
            local_imprint(0.1, 1e-6)

    def test_unresolvable_tc_raises(self):
        # The power law's inverse rounds onto lambda_c at and below 6.173098052100439e-06 K.
        with pytest.raises(ValueError, match="resolvable"):
            lambda_from_tc(6.173098052100439e-06)
        lam = lambda_from_tc(6.17309805210044e-06)
        assert lam > POWER_LAW.lambda_c and tc_from_lambda(lam) > 0.0
        with pytest.raises(ValueError, match="resolvable"):
            lambda_from_tc(5.0, SfPairingMap(exponent=0.001))
        with pytest.raises(ValueError):
            lambda_from_tc(1e-323, MCMILLAN)  # 1.45 Tc / omega underflows to 0
        assert tc_from_lambda(lambda_from_tc(1e-300, MCMILLAN), MCMILLAN) > 0.0

    def test_real_scalars_of_any_numeric_type_pass(self):
        phi = np.radians([0.0, 27.5])
        cfg = replace(DEFAULT, gamma_imp=np.float64(27.0), hotspot_peak=31, nu=np.int64(12))
        np.testing.assert_array_equal(
            scattering_rate(phi, 40.0, cfg),
            scattering_rate(phi, 40.0, replace(DEFAULT, hotspot_peak=31.0)),
        )
        assert lambda_from_tc(np.float64(13.5)) == lambda_from_tc(13.5)
        assert glue_gap_scale(SfGlueConfig(tc_target=np.int64(5))) == glue_gap_scale(
            SfGlueConfig(tc_target=5.0)
        )

    def test_invalid_quadrature_raises(self):
        with pytest.raises(ValueError):
            quadrature_resistivity(-1.0, 1.0)
        with pytest.raises(ValueError):
            quadrature_field_derivative(10.0, np.nan)
        with pytest.raises(ValueError):
            quadrature_resistivity(10.0, 1.0, SfQuadratureConfig(alpha_kb=0.0))
        with pytest.raises(ValueError):
            quadrature_resistivity(10.0, 1.0, SfQuadratureConfig(rho0=-1.0))
        with pytest.raises(ValueError):
            quadrature_resistivity(10.0, 1.0, SfQuadratureConfig(rho0="15.5"))

    def test_module_tuples(self):
        assert SF_INELASTIC_FORMS == ("afm_boundary", "isotropic", "antinodal", "none")
        assert SF_PAIRING_FORMS == ("power_law", "mcmillan")
        assert SF_COHERENCE_MODELS == ("clean", "dirty", "local")
        assert SF_DOPING_REGIMES == ("fs_reconstruction", "superconducting", "beyond_dome")
        assert hash(DEFAULT) == hash(SfConfig()) and hash(SfGlueConfig()) == hash(SfGlueConfig())

    def test_parts_b_to_d_docstrings_say_speculative(self):
        lines, _ = inspect.getsourcelines(spin_fluctuation)
        part_b = next(i for i, line in enumerate(lines, 1) if line.startswith("# Part B"))
        public = {
            name: obj
            for name, obj in vars(spin_fluctuation).items()
            if not name.startswith("_")
            and (inspect.isfunction(obj) or inspect.isclass(obj))
            and obj.__module__ == spin_fluctuation.__name__
            and inspect.getsourcelines(obj)[1] > part_b
        }
        assert {"tc_ceiling", "moire_profile", "SfQuadratureConfig"} <= public.keys()
        assert [name for name, obj in public.items() if "SPECULATIVE" not in obj.__doc__] == []


class TestCrossStackParityAnchors:
    """SF-amendments §9.3 v2, asserted identically in crates/moire-core/src/spin_fluctuation.rs.

    Algorithm-parity (AP) rows pin the N = 1024 per-cell integrator's own value, which
    sits 6e-6 to 5.2e-4 below the converged one (see TestAccuracy).
    """

    def test_p1_fermi_surface_point(self):
        kf, vf, tilt, _, _ = fermi_surface_at(0.3)
        assert float(kf) == _rel(5614221434.684519, 1e-14)
        assert float(vf) == _rel(499330.6678195376, 1e-14)
        assert float(tilt) == _rel(0.19556905668726163, 1e-14)

    def test_p1b_to_p1d(self):
        assert luttinger_holes() == _rel(0.8391564248900439, 1e-12)
        assert zone_boundary_crossing_deg() == _rel(27.196569908994878, 1e-12)
        assert drude_hall_coefficient() == _rel(7.403312411077447e-10, 1e-12)

    def test_p2_scattering_rates(self):
        phi = np.radians([27.5, 45.0, 0.0])
        want = (106.3201227826564, 35.74824703294329, 27.12151850594606)
        np.testing.assert_allclose(scattering_rate(phi, 40.0), want, rtol=1e-13, atol=0)
        phi_b = np.radians([0.0, 10.0, 27.5])
        np.testing.assert_allclose(
            scattering_rate(phi_b, 40.0, CASE_C), (74.92, 69.31442485713072, 74.16519736591698),
            rtol=1e-13, atol=0,
        )  # fmt: skip
        np.testing.assert_allclose(
            scattering_rate(phi_b, 40.0, replace(DEFAULT, inelastic="isotropic")),
            (74.92, 74.92, 106.32), rtol=1e-13, atol=0,
        )  # fmt: skip

    def test_p3_zero_field_resistivity(self):
        assert zero_field_resistivity(DEFAULT, [40.0])[0] == _rel(24.49517992890017, 1e-12)

    def test_p4_magnetoresistance_ap(self):
        rho = resistivity_tensor(DEFAULT, 40.0, [0.0, 20.0])
        assert rho[1, 0] == _rel(25.04265480287949, 1e-12)
        assert rho[1, 1] == _rel(1.192527321423354, 1e-12)
        assert rho[1, 0] / rho[0, 0] - 1.0 == _rel(0.022350310369975812, 1e-11)
        assert _mr(DEFAULT, 20.0, 35.0) == _rel(0.041376414577780185, 1e-11)

    def test_p5_high_field_slope_ap(self):
        assert high_field_slope(DEFAULT, 40.0) == _rel(0.05026085857440421, 1e-9)

    def test_p6_hall(self):
        assert hall_coefficient(CASE_A, 0.0, cvc=False) == _rel(7.339685060373182e-10, 1e-12)
        assert hall_coefficient(CASE_A, 0.0, cvc=True) == _rel(5.309220240734655e-10, 1e-12)
        assert hall_coefficient(DEFAULT, 40.0, cvc=False) == _rel(6.131175356207996e-10, 1e-12)
        assert hall_coefficient(DEFAULT, 40.0, cvc=True) == _rel(5.092606357247605e-10, 1e-12)

    def test_p6b_low_field_chambers_ap(self):
        assert hall_coefficient_chambers(DEFAULT, 40.0, 0.1) == _rel(6.131011216626595e-10, 1e-11)

    def test_p6c_hall_with_cvc(self):
        assert hall_coefficient(CASE_C, 50.0, cvc=True) == _rel(1.9749077940876073e-09, 1e-12)
        assert hall_coefficient(CASE_B, 0.0, cvc=True) == _rel(4.928676220661514e-10, 1e-12)

    def test_p6d_cvc_geometry(self):
        centres = np.degrees(cvc_centres()[:2])
        assert centres[0] == _rel(27.196569908994878, 1e-12)
        assert centres[1] == _rel(62.80343009100512, 1e-12)
        assert abs(cvc_partner_angle(0.0) - 3.141592653589793) <= 1e-15

    def test_p6e_hall_ratio(self):
        assert hall_ratio_to_case_a(DEFAULT, 40.0) == _rel(0.9592004336484117, 1e-12)

    def test_p7_couplings(self):
        assert lambda_tr() == _rel(0.41758273992325223, 1e-12)
        assert lambda_tr(replace(DEFAULT, doping=0.116)) == _rel(2.714287809501139, 1e-12)
        assert lambda_tr(replace(DEFAULT, inelastic="isotropic")) == _rel(0.9117468534052734, 1e-12)
        assert lambda_tr(CASE_C) == _rel(0.4475272735616924, 1e-12)
        assert drude_resistivity_per_rate() == _rel(5.5005672382590704e-21, 1e-12)
        for x, want in (
            (0.159, 0.4352438971760371),
            (0.116, 1.491881542660157),
            (0.188, 0.2946327374251149),
        ):
            assert lambda_resistive(replace(DEFAULT, doping=x)) == _rel(want, 1e-9)

    def test_p8_pairing_maps(self):
        assert (POWER_LAW.lambda_c, POWER_LAW.exponent, POWER_LAW.lambda_ref) == (
            0.30462015776911294, 0.3926512381199364, 2.1714302476009117,
        )  # fmt: skip
        assert (MCMILLAN.omega_k, MCMILLAN.mu_star) == (178.04201155099287, 0.0)
        lam = lambda_tr(replace(DEFAULT, doping=0.116))
        assert tc_from_lambda(lam) == _rel(27.635552408315235, 1e-12)
        assert tc_from_lambda(lam, MCMILLAN) == _rel(29.58591451200863, 1e-12)

    def test_p9_operating_points(self):
        lam = lambda_from_tc(13.5)
        assert lam == _rel(0.6932743989110509, 1e-12)
        assert log_sensitivity(lam) == _rel(0.700404169756287, 1e-12)
        lam_m = lambda_from_tc(13.5, MCMILLAN)
        assert lam_m == _rel(0.8905891525900327, 1e-12)
        assert log_sensitivity(lam_m, MCMILLAN) == _rel(1.16776630051629, 1e-12)
        assert equivalent_doping(lam) == _rel(0.1435584333754864, 1e-12)
        assert equivalent_doping(POWER_LAW.lambda_c) == _rel(0.17052577242235883, 1e-12)

    @pytest.mark.parametrize(
        ("tc", "coherence", "want"),
        [
            (13.5, "clean", (2.9569962300271597, 3.09, 3.2092567693397047)),
            (13.5, "dirty", (2.9569962300271597, 3.09, 3.2092567693397047)),
            (13.5, "local", (2.860581453791304, 3.09, 3.295706309325275)),
            (5.0, "clean", (0.8818228883664684, 1.1444444444444444, 1.2321772587067223)),
            (5.0, "dirty", (0.6771928153962904, 1.1444444444444444, 1.300537116337901)),
            (5.0, "local", (0.0, 1.1444444444444444, 1.5267639058679714)),
            (2.0, "clean", (0.4462301216137267, 0.45777777777777773, 0.4760946886571619)),
            (2.0, "dirty", (0.34672682120871845, 0.45777777777777773, 0.6339269846513088)),
            (2.0, "local", (0.0, 0.45777777777777773, 1.1839056290920276)),
        ],
    )
    def test_p10_glue_gaps(self, tc, coherence, want):
        gap = glue_modulated_gap(
            np.array([0.0, 0.5, 1.0]), SfGlueConfig(tc_target=tc, coherence=coherence)
        ).gap
        for value, expected in zip(gap, want, strict=True):
            if expected == 0.0:
                assert (
                    value == 0.0
                )  # lambda <= lambda_c: Tc = 0, Delta = Delta_bar + 1 (0 - Delta_bar)
            else:
                assert value == _rel(expected, 1e-12)

    @pytest.mark.parametrize(
        ("tc", "want"),
        [
            (13.5, (20.0, 20.0, 0.5797428855287498, 0.5797428855287498, 3.09)),
            (
                5.0,
                (
                    54.0,
                    32.86335345030997,
                    0.22947514608755187,
                    0.40827812246926076,
                    1.1444444444444444,
                ),
            ),
            (
                2.0,
                (
                    135.0,
                    51.96152422706632,
                    0.02522546249428618,
                    0.24258704104891599,
                    0.45777777777777773,
                ),
            ),
        ],
    )
    def test_p10b_coherence(self, tc, want):
        clean = SfGlueConfig(tc_target=tc)
        dirty = replace(clean, coherence="dirty")
        got = (coherence_length(clean), coherence_length(dirty), coherence_factor(clean),
               coherence_factor(dirty), glue_gap_scale(clean))  # fmt: skip
        for value, expected in zip(got, want, strict=True):
            assert value == _rel(expected, 1e-14)

    @pytest.mark.parametrize(
        ("tc", "want"),
        [
            (
                13.5,
                (0.04081885749393931, 0.04081885749393931, 0.0704085526753999, 0.07004041697562871),
            ),
            (
                5.0,
                (0.1530674433525382, 0.27233488876089784, 0.6670327744083371, 0.4254496324492375),
            ),
            (2.0, (0.03261906594549964, 0.313689498905742, 1.293100808474302, 4.022759727552418)),
        ],
    )
    def test_p10c_imprints(self, tc, want):
        clean, local, linear = imprint_curve(SfGlueConfig(), [tc])
        dirty = imprint_curve(SfGlueConfig(coherence="dirty"), [tc])[0]
        for value, expected in zip((clean[0], dirty[0], local[0], linear[0]), want, strict=True):
            assert value == _rel(expected, 1e-12)

    def test_p10d_to_p10f(self):
        assert SfGlueConfig().moire_period == _rel(36.68576576576573, 1e-15)
        assert fragmentation_tc() == _rel(5.177318594015492, 1e-12)
        assert linearized_glue_amplitude() == _rel(0.12547078933296135, 1e-12)

    def test_p11_quadrature(self):
        assert quadrature_field_derivative(40.0, 20.0) == _rel(0.18569533817705186, 1e-13)
        assert quadrature_resistivity(40.0, 20.0) == _rel(30.208131845707605, 1e-13)

    def test_p12_resistivity_fit(self):
        fit = resistivity_fit(DEFAULT)
        assert fit.rho0 == _rel(15.80541402181645, 1e-9)
        assert fit.a1 == _rel(0.19693692765571835, 1e-9)
        assert fit.a2 == _rel(0.0005046269609317277, 1e-9)

    def test_p13_near_optimal_doping(self):
        cfg = replace(DEFAULT, doping=0.116)
        rho = resistivity_tensor(cfg, 40.0, [0.0, 20.0])[:, 0]
        assert rho[0] == _rel(41.96086725955268, 1e-12)
        assert rho[1] / rho[0] - 1.0 == _rel(0.03951909517654206, 1e-11)  # AP


class TestAccuracy:
    """SF-amendments §9.3b: converged values (independent spectral solver, N = 4096)."""

    def test_anchored_quantities(self):
        rho = resistivity_tensor(DEFAULT, 40.0, [0.0, 20.0])
        got = {
            "rho_xx(20 T, 40 K)": (rho[1, 0], 25.04281275669245),
            "rho_yx(20 T, 40 K)": (rho[1, 1], 1.1926091496601852),
            "MR(20 T, 40 K)": (rho[1, 0] / rho[0, 0] - 1.0, 0.022356758733017124),
            "MR(35 T, 20 K)": (_mr(DEFAULT, 20.0, 35.0), 0.04139795183670203),
            "gamma_1(33 T, 40 K)": (high_field_slope(DEFAULT, 40.0), 0.05027355287307245),
            "MR(20 T, 40 K), x = 0.116": (
                _mr(replace(DEFAULT, doping=0.116), 40.0, 20.0),
                0.03953104183231093,
            ),
            "R_H Chambers 0.1 T, 40 K": (
                hall_coefficient_chambers(DEFAULT, 40.0, 0.1),
                6.131167143845524e-10,
            ),
        }
        for name, (value, converged) in got.items():
            assert value == _rel(converged, 1e-3), name

    def test_documented_family(self):
        converged = {  # MR(20 T), MR(35 T), (rho(33.5 T) - rho(32.5 T)) / 1 T
            20.0: (0.01951457012176694, 0.04139795183669137, 0.02762606147097202),
            30.0: (0.020448676521295228, 0.046647886040358744, 0.03843795728723265),
            40.0: (0.022356758733012017, 0.052521356592582835, 0.050273552873687066),
            50.0: (0.024721219355334556, 0.058467174795128196, 0.06263202760688458),
        }
        fam = _family(DEFAULT)
        for k, t in enumerate(MR_TEMPS):
            mr20, mr35, slope = converged[t]
            assert fam.mr[k, I20] == _rel(mr20, 1e-3)
            assert fam.mr[k, I35] == _rel(mr35, 1e-3)
            assert fam.rho_xx[k, 67] - fam.rho_xx[k, 65] == _rel(slope, 1e-3)

    def test_kohler_collapse_value(self):
        rho0 = float(zero_field_resistivity(KOHLER, [20.0])[0])
        assert _mr(KOHLER, 20.0, rho0) == _rel(0.00151617326, 1e-3)


class TestPerformance:
    def test_magnetoresistance_at_defaults(self):
        magnetoresistance(DEFAULT)  # warm-up
        start = time.perf_counter()
        fam = magnetoresistance(DEFAULT)
        elapsed = time.perf_counter() - start
        print(f"\nmagnetoresistance(SfConfig()), 4 x 71: {elapsed * 1e3:.1f} ms (target 100 ms)")
        assert fam.rho_xx.shape == (4, 71)
        assert elapsed < 1.0  # non-flaky guard against a per-field Python loop
