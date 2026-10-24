"""Tests for the pair-density-wave (PDW) modality model (SPECULATIVE)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from waytogocoop.computation.moire import (
    estimate_moire_period,
    moire_periodicity_1d,
    moire_periodicity_with_twist,
)
from waytogocoop.computation.pdw import (
    PDW_REGIMES,
    PdwConfig,
    charge_spectrum,
    classify_regime,
    compute_pdw_state,
    critical_field_along,
    dvector_overlaps,
    field_sweep,
    harmonic_vectors,
    landau_amplitude,
    moire_locked_config,
    moire_locked_pdw_ratio,
    moire_locked_period,
    moire_locked_wavevectors,
    onset_field_along,
    pdw_fields_at,
    pdw_peaks,
    pdw_real_space,
    reduced_field,
    regime_map,
    temperature_sweep,
    ute2_wavevectors,
)
from waytogocoop.config import A_TE_UTE2, TC_FETE_FILM

Q_TE = 2.0 * np.pi / A_TE_UTE2
X_AXIS, Y_AXIS, Z_AXIS = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)


def _state(**overrides):
    return compute_pdw_state(PdwConfig(**overrides))


def _visible(**overrides) -> set[str]:
    config = PdwConfig(**overrides)
    return {p.label for p in pdw_peaks(config, compute_pdw_state(config)) if p.visible}


class TestWavevectors:
    def test_ute2_p3_magnitude(self):
        p = ute2_wavevectors()
        assert np.linalg.norm(p[2]) / Q_TE == pytest.approx(0.28, rel=1e-12)

    def test_ute2_p1_p2_mirror_partners(self):
        p = ute2_wavevectors()
        np.testing.assert_allclose(p[0], [-p[1][0], p[1][1]])

    def test_harmonics_match_table_1(self):
        p = ute2_wavevectors()
        q = 2.0 * p
        h = harmonic_vectors(p)
        np.testing.assert_allclose(h[0], q[2] - q[1])
        np.testing.assert_allclose(h[1], q[2] - q[0])

    def test_h1_h2_p3_close_a_hexagonal_triple(self):
        # Zhu et al.: h1, h2 and p3 share |k| ~ 0.28 q_Te and form a
        # hexagonal-like modulation; the mirror-consistent h2 sums them to 0.
        p = ute2_wavevectors()
        h = harmonic_vectors(p)
        np.testing.assert_allclose(h.sum(axis=0) + p[2], [0.0, 0.0], atol=1e-12)
        for k in (*h, p[2]):
            assert np.linalg.norm(k) / Q_TE == pytest.approx(0.28, abs=0.02)

    def test_moire_locked_magnitudes_and_angles(self):
        p = moire_locked_wavevectors(36.0)
        np.testing.assert_allclose(np.linalg.norm(p, axis=1), 2.0 * np.pi / 36.0)
        cosines = [p[i] @ p[(i + 1) % 3] / np.linalg.norm(p[0]) ** 2 for i in range(3)]
        np.testing.assert_allclose(cosines, -0.5, atol=1e-12)

    def test_moire_locked_twist_rotates(self):
        p0 = moire_locked_wavevectors(36.0)
        p1 = moire_locked_wavevectors(36.0, twist_angle_deg=90.0)
        np.testing.assert_allclose(p1[:, 0], -p0[:, 1], atol=1e-12)
        np.testing.assert_allclose(p1[:, 1], p0[:, 0], atol=1e-12)

    @pytest.mark.parametrize("period", [0.0, -5.0, np.inf, np.nan])
    def test_moire_locked_invalid_period(self, period):
        with pytest.raises(ValueError):
            moire_locked_wavevectors(period)


class TestLandauAmplitudes:
    def test_full_amplitude_at_zero(self):
        assert landau_amplitude(0.0, 5.0, 0.0) == 1.0

    def test_vanishes_at_transition(self):
        assert landau_amplitude(5.0, 5.0, 0.0) == 0.0
        assert landau_amplitude(6.0, 5.0, 0.0) == 0.0

    def test_squared_amplitude_linear_in_t_and_h(self):
        assert landau_amplitude(1.0, 4.0, 0.25) ** 2 == pytest.approx(0.5, rel=1e-12)

    def test_reduced_field_is_ellipsoidal(self):
        assert reduced_field((3.0, 0.0, 0.0), (6.0, 1.0, 1.0)) == pytest.approx(0.5)
        assert reduced_field((3.0, 4.0, 0.0), (10.0, 10.0, 10.0)) == pytest.approx(0.5)

    def test_critical_field_zeroes_the_amplitude(self):
        cfg = PdwConfig()
        for u in (X_AXIS, Y_AXIS, Z_AXIS, (1.0, 1.0, 1.0)):
            h_star = critical_field_along(cfg.t_pdw, cfg.h_pdw, 1.0, u)
            b = h_star * np.asarray(u) / np.linalg.norm(u)
            assert landau_amplitude(1.0, cfg.t_pdw, reduced_field(b, cfg.h_pdw)) == (
                pytest.approx(0.0, abs=1e-7)
            )

    def test_critical_field_zero_above_transition(self):
        assert critical_field_along(2.1, (20.0, 11.0, 34.0), 3.0, Z_AXIS) == 0.0

    @pytest.mark.parametrize(
        ("delta0", "delta_p", "regime"),
        [
            (0.5, 0.5, "coexistence"),
            (0.0, 0.5, "pdw_only"),
            (0.5, 0.0, "uniform_only"),
            (0.0, 0.0, "normal"),
        ],
    )
    def test_classify_regime(self, delta0, delta_p, regime):
        assert classify_regime(delta0, delta_p) == regime
        assert regime in PDW_REGIMES


class TestRegimes:
    def test_coexistence_below_tc(self):
        assert _state(temperature=1.0).regime == "coexistence"

    def test_cheshire_cat_window_between_tc_and_t_pdw(self):
        state = _state(temperature=3.0)
        assert state.regime == "pdw_only"
        assert state.delta0 == 0.0
        assert state.delta_p > 0.0

    def test_normal_above_t_pdw(self):
        assert _state(temperature=5.5).regime == "normal"

    def test_uniform_only_when_field_kills_pdw_first(self):
        # Along a the PDW critical field (3.75 T) is below Hc2 (11 T).
        assert _state(temperature=0.3, field=(0.0, 5.0, 0.0)).regime == "uniform_only"

    def test_no_pdw_only_window_when_t_pdw_equals_tc(self):
        cfg = PdwConfig(t_pdw=2.1)
        sweep = temperature_sweep(cfg, np.linspace(0.0, 3.0, 61))
        pdw_only = (sweep.delta_p > 0) & (sweep.delta0 == 0)
        assert not np.any(pdw_only)


@pytest.mark.speculative
class TestCompositeOrders:
    def test_p_cdw_needs_uniform_superconductivity(self):
        assert np.all(np.abs(_state(temperature=3.0).rho_p) == 0.0)
        assert abs(_state(temperature=1.0).rho_p[2]) > 0.0

    def test_q_cdw_is_pdw_alone(self):
        state = _state(temperature=3.0)
        np.testing.assert_allclose(np.abs(state.rho_q), state.delta_p**2)

    def test_q_continuous_across_tc_without_harmonic_feed(self):
        below = _state(temperature=2.1 - 1e-9).rho_q
        above = _state(temperature=2.1 + 1e-9).rho_q
        np.testing.assert_allclose(np.abs(below), np.abs(above), rtol=1e-6)

    def test_harmonic_feed_enhances_q_below_tc_only(self):
        base = _state(temperature=1.0)
        fed = _state(temperature=1.0, kappa_harm=0.5)
        assert abs(fed.rho_q[2]) > abs(base.rho_q[2])
        np.testing.assert_allclose(
            np.abs(_state(temperature=3.0, kappa_harm=0.5).rho_q),
            np.abs(_state(temperature=3.0).rho_q),
        )

    def test_harmonic_is_product_of_q_amplitudes(self):
        state = _state(temperature=1.0)
        assert abs(state.rho_h[0]) == pytest.approx(abs(state.rho_q[2] * state.rho_q[1]))
        assert abs(state.rho_h[1]) == pytest.approx(abs(state.rho_q[2] * state.rho_q[0]))

    def test_sliding_phase_doubles_for_q(self):
        state = _state(temperature=1.0, phases=(0.0, 0.0, 0.3))
        assert np.angle(-state.rho_p[2]) == pytest.approx(0.3)
        assert np.angle(-state.rho_q[2]) == pytest.approx(0.6)

    def test_pi_phase_flips_p_but_not_q(self):
        ref = _state(temperature=1.0)
        flipped = _state(temperature=1.0, phases=(0.0, 0.0, np.pi))
        assert flipped.rho_p[2] == pytest.approx(-ref.rho_p[2])
        assert flipped.rho_q[2] == pytest.approx(ref.rho_q[2])


@pytest.mark.speculative
class TestFieldInducedDVectorRotation:
    def test_zero_field_overlaps(self):
        np.testing.assert_allclose(_state().overlap, [0.0, 0.0, 1.0])

    def test_overlap_formula_above_onset(self):
        c = dvector_overlaps((0.0, 0.0, 9.0), (0.4, 0.4, 4.5), (0.0, 0.0, 1.0))
        np.testing.assert_allclose(c, [np.sqrt(0.75), np.sqrt(0.75), 1.0])

    def test_overlap_saturates_at_high_field(self):
        c = dvector_overlaps((0.0, 0.0, 1.0e4), (0.4, 0.4, 4.5), (0.0, 0.0, 1.0))
        np.testing.assert_allclose(c, 1.0, atol=1e-6)

    def test_onset_fields_along_axes(self):
        cfg = PdwConfig()
        assert onset_field_along(cfg.h_rot, X_AXIS) == pytest.approx(0.4)
        assert onset_field_along(cfg.h_rot, Y_AXIS) == pytest.approx(0.4)
        assert onset_field_along(cfg.h_rot, Z_AXIS) == pytest.approx(4.5)

    def test_sdw_weight_converts_into_p_cdw(self):
        for bz in (0.0, 4.0, 6.0, 12.0):
            state = _state(field=(0.0, 0.0, bz))
            total = (2.0 * state.delta0 * state.delta_p) ** 2
            np.testing.assert_allclose(
                np.abs(state.rho_p) ** 2 + np.abs(state.spin_p) ** 2, total
            )

    def test_moire_locked_has_no_rotation_physics(self):
        cfg = replace(moire_locked_config(36.0), field=(5.0, 5.0, 5.0))
        state = compute_pdw_state(cfg)
        np.testing.assert_allclose(state.overlap, 1.0)
        np.testing.assert_allclose(np.abs(state.spin_p), 0.0)


@pytest.mark.speculative
class TestZhuObservations:
    """The UTe2 defaults against Zhu et al., PNAS 123, e2602117123 (2026)."""

    def test_q_suppression_fields_at_4p2_k_inside_measured_windows(self):
        cfg = PdwConfig()
        windows = {X_AXIS: (1.5, 3.0), Y_AXIS: (0.4, 0.8), Z_AXIS: (4.0, 8.0)}
        h_sup = {}
        for axis, (lo, hi) in windows.items():
            h_sup[axis] = critical_field_along(cfg.t_pdw, cfg.h_pdw, 4.2, axis)
            assert lo < h_sup[axis] < hi
        # H_sup^b* > H_sup^c* > H_sup^a, mirroring Hc2^b* > Hc2^c* > Hc2^a.
        assert h_sup[Z_AXIS] > h_sup[X_AXIS] > h_sup[Y_AXIS]
        hc2 = {axis: critical_field_along(cfg.tc, cfg.hc2, 0.0, axis) for axis in windows}
        assert hc2[Z_AXIS] > hc2[X_AXIS] > hc2[Y_AXIS]

    def test_zero_field_peaks_at_300_mk(self):
        assert _visible() == {"p3", "q1", "q2", "q3", "h1", "h2"}

    def test_p3_vanishes_at_tc(self):
        assert "p3" in _visible(temperature=2.0)
        assert "p3" not in _visible(temperature=2.15)

    def test_h_vanishes_near_3p5_k(self):
        assert {"h1", "h2"} <= _visible(temperature=3.3)
        assert not {"h1", "h2"} & _visible(temperature=3.6)

    def test_q_survives_to_4p2_k_then_fades(self):
        assert {"q1", "q2", "q3"} <= _visible(temperature=4.2)
        assert not {"q1", "q2", "q3"} & _visible(temperature=4.8)

    @pytest.mark.parametrize(
        ("below", "above"),
        [((0.0, 0.0, 4.0), (0.0, 0.0, 5.0)), ((0.0, 0.3, 0.0), (0.0, 0.5, 0.0))],
    )
    def test_p2_is_field_induced_at_300_mk(self, below, above):
        assert "p2" not in _visible(field=below)
        assert {"p1", "p2"} <= _visible(field=above)

    def test_p2_field_window_shrinks_at_1p6_k(self):
        assert "p2" in _visible(temperature=1.6, field=(0.0, 0.0, 5.0))
        assert "p2" in _visible(temperature=1.6, field=(0.0, 0.0, 6.0))
        assert "p2" not in _visible(temperature=1.6, field=(0.0, 0.0, 8.0))
        for bz in (5.0, 6.0, 8.0):
            assert "p2" not in _visible(temperature=2.5, field=(0.0, 0.0, bz))

    def test_p2_strengthens_to_2_t_along_c_star(self):
        def p2(bx):
            return abs(_state(field=(bx, 0.0, 0.0)).rho_p[1]) ** 2

        assert p2(2.0) > p2(0.5)
        assert p2(2.0) > p2(3.0)

    def test_q_field_sensitivity_grows_with_temperature(self):
        assert {"q1", "q2", "q3"} <= _visible(temperature=0.3, field=(0.0, 0.8, 0.0))
        assert not {"q1", "q2", "q3"} & _visible(temperature=4.2, field=(0.0, 0.8, 0.0))


class TestRealSpace:
    def test_field_shapes(self):
        x = np.linspace(-40.0, 40.0, 50)
        y = np.linspace(-30.0, 30.0, 40)
        cfg = PdwConfig()
        fields = pdw_real_space(cfg, compute_pdw_state(cfg), x, y)
        for arr in (fields.pair_amplitude, fields.charge_density, fields.spin_density):
            assert arr.shape == (40, 50)
            assert np.all(np.isfinite(arr))

    def test_pair_amplitude_at_origin(self):
        cfg = PdwConfig()
        state = compute_pdw_state(cfg)
        delta0 = cfg.delta0_mev * state.delta0
        delta_p = cfg.delta0_mev * cfg.pdw_ratio * state.delta_p
        # p3 is parallel to d0, p1 and p2 orthogonal at zero field.
        expected = np.hypot(delta0 + 2.0 * delta_p, 4.0 * delta_p)
        pair, _charge, _spin = pdw_fields_at(cfg, state, 0.0, 0.0)
        assert float(pair) == pytest.approx(expected, rel=1e-12)

    def test_pair_amplitude_nonnegative(self):
        x = np.linspace(-50.0, 50.0, 64)
        cfg = PdwConfig(temperature=3.0)
        fields = pdw_real_space(cfg, compute_pdw_state(cfg), x, x)
        assert np.all(fields.pair_amplitude >= 0.0)

    def test_normal_state_fields_vanish(self):
        x = np.linspace(-20.0, 20.0, 16)
        cfg = PdwConfig(temperature=6.0)
        fields = pdw_real_space(cfg, compute_pdw_state(cfg), x, x)
        assert np.all(fields.pair_amplitude == 0.0)
        assert np.all(fields.charge_density == 0.0)

    def test_moire_what_if_halves_the_charge_period_above_tc(self):
        # Above Tc an intrinsic PDW leaves only q = 2Q_m (and h) charge order:
        # the plane-wave projection at p3 vanishes, the one at q3 does not.
        period = 36.0
        cfg = replace(moire_locked_config(period), t_pdw=1.5 * TC_FETE_FILM)
        state = compute_pdw_state(replace(cfg, temperature=1.2 * TC_FETE_FILM))
        assert state.regime == "pdw_only"
        x = np.linspace(-15.0 * period, 15.0 * period, 601)
        X, Y = np.meshgrid(x, x)
        _pair, charge, _spin = pdw_fields_at(cfg, state, X, Y)
        p3 = np.asarray(cfg.p_vectors[2])

        def projection(k):
            return abs(np.mean(charge * np.exp(-1j * (k[0] * X + k[1] * Y))))

        assert projection(2.0 * p3) == pytest.approx(abs(state.rho_q[2]), rel=0.05)
        assert projection(p3) < 0.01 * projection(2.0 * p3)


class TestSweeps:
    def test_temperature_sweep_shapes_and_physics(self):
        temps = np.linspace(0.0, 6.0, 121)
        sweep = temperature_sweep(PdwConfig(), temps)
        assert sweep.intensity_p.shape == (3, 121)
        assert sweep.intensity_q.shape == (3, 121)
        assert sweep.intensity_h.shape == (2, 121)
        assert np.all(sweep.intensity_p[:, temps >= 2.1] == 0.0)
        assert np.all(sweep.delta_p[temps >= 5.0] == 0.0)
        assert np.all(np.diff(sweep.intensity_q[2]) <= 1e-15)

    def test_field_sweep_direction_is_normalised(self):
        mags = np.linspace(0.0, 10.0, 11)
        a = field_sweep(PdwConfig(), mags, (0.0, 0.0, 1.0))
        b = field_sweep(PdwConfig(), mags, (0.0, 0.0, 7.0))
        np.testing.assert_allclose(a.intensity_p, b.intensity_p)

    def test_empty_sweeps_keep_their_layout(self):
        sweep = temperature_sweep(PdwConfig(), [])
        assert sweep.intensity_p.shape == (3, 0)
        assert sweep.intensity_q.shape == (3, 0)
        assert sweep.intensity_h.shape == (2, 0)

    def test_field_sweep_zero_direction_raises(self):
        with pytest.raises(ValueError):
            field_sweep(PdwConfig(), [0.0, 1.0], (0.0, 0.0, 0.0))

    def test_regime_map_matches_state(self):
        cfg = PdwConfig()
        temps = np.linspace(0.0, 6.0, 13)
        mags = np.linspace(0.0, 12.0, 7)
        regimes = regime_map(cfg, temps, mags, Y_AXIS)
        assert regimes.shape == (13, 7)
        for i, t in enumerate(temps):
            for j, b in enumerate(mags):
                state = _state(temperature=t, field=(0.0, b, 0.0))
                assert PDW_REGIMES[regimes[i, j]] == state.regime


class TestMoireLocked:
    def test_wavevector_follows_locked_period(self, fete, sb2te3):
        period = moire_locked_period(fete.a, sb2te3.a)
        cfg = moire_locked_config(period)
        np.testing.assert_allclose(
            np.linalg.norm(np.asarray(cfg.p_vectors), axis=1), 2.0 * np.pi / period
        )

    def test_period_reduces_to_both_limits(self, fete, sb2te3):
        # theta = 0 is the mismatch formula (and the viewer's estimate);
        # equal lattices give the twist formula.
        mismatch = moire_periodicity_1d(fete.a, sb2te3.a)
        assert moire_locked_period(fete.a, sb2te3.a) == pytest.approx(mismatch, rel=1e-12)
        assert moire_locked_period(fete.a, sb2te3.a) == pytest.approx(
            estimate_moire_period(fete.a, sb2te3.a), rel=1e-12
        )
        assert moire_locked_period(2.46, 2.46, 1.0) == pytest.approx(
            moire_periodicity_with_twist(2.46, 1.0), rel=1e-9
        )
        assert np.isinf(moire_locked_period(2.46, 2.46, 0.0))

    def test_twist_shortens_a_mismatched_period_only_slightly(self, fete, sb2te3):
        # An 11% mismatch dominates a 1 deg twist; the twist formula alone
        # (231.6 Å with a_eff) would be off by more than 6x.
        period = moire_locked_period(fete.a, sb2te3.a, 1.0)
        assert 35.0 < period < moire_periodicity_1d(fete.a, sb2te3.a)

    def test_ratio_is_half_the_gap_split_times_cpdm_suppression(self):
        ratio = moire_locked_pdw_ratio(40.0, coherence_length=20.0, delta_1=2.0, delta_2=4.0)
        assert ratio == pytest.approx(0.5 * 1.0 / 3.0 * np.exp(-0.5), rel=1e-12)

    @pytest.mark.parametrize(
        ("period", "xi"), [(0.0, 20.0), (np.nan, 20.0), (36.0, 0.0), (36.0, -1.0)]
    )
    def test_ratio_rejects_bad_inputs(self, period, xi):
        with pytest.raises(ValueError):
            moire_locked_pdw_ratio(period, coherence_length=xi)

    def test_period_rejects_bad_lattices(self):
        for a1, a2 in ((0.0, 4.0), (np.nan, 4.0), (3.8, -1.0)):
            with pytest.raises(ValueError):
                moire_locked_period(a1, a2)

    def test_imprinted_cpdm_has_no_cheshire_cat_window(self):
        cfg = moire_locked_config(36.0)
        temps = np.linspace(0.0, 1.5 * TC_FETE_FILM, 91)
        sweep = temperature_sweep(cfg, temps)
        above = temps >= cfg.tc
        assert np.all(sweep.intensity_q[:, above] == 0.0)

    def test_what_if_intrinsic_pdw_opens_the_window(self):
        cfg = replace(moire_locked_config(36.0), t_pdw=1.5 * TC_FETE_FILM)
        state = compute_pdw_state(replace(cfg, temperature=1.2 * TC_FETE_FILM))
        assert state.regime == "pdw_only"
        assert np.all(np.abs(state.rho_q) > 0.0)


class TestChargeSpectrum:
    def test_cropped_to_the_composite_peaks(self):
        cfg = PdwConfig()
        spectrum = charge_spectrum(cfg, compute_pdw_state(cfg), 40.0)
        k_max = max(np.hypot(p.kx, p.ky) for p in pdw_peaks(cfg, compute_pdw_state(cfg)))
        assert np.abs(spectrum["kx"]).max() <= 1.3 * k_max
        assert spectrum["power_spectrum"].shape == (len(spectrum["ky"]), len(spectrum["kx"]))

    def test_q3_peak_stands_out_above_tc(self):
        # Between Tc and T_PDW the spectrum is bright at q3 = 2 p3 and dark
        # at p3, where the p CDW has died with the uniform condensate.
        cfg = PdwConfig(temperature=3.0)
        spectrum = charge_spectrum(cfg, compute_pdw_state(cfg), 40.0)
        kx, ky, power = spectrum["kx"], spectrum["ky"], spectrum["power_spectrum"]
        p3 = np.asarray(cfg.p_vectors[2])

        def at(k):
            return power[np.argmin(np.abs(ky - k[1])), np.argmin(np.abs(kx - k[0]))]

        assert at(2.0 * p3) > at(p3) + 3.0  # three decades in log10 power

    def test_resolution_independent_of_display_grid(self):
        cfg = PdwConfig()
        state = compute_pdw_state(cfg)
        a = charge_spectrum(cfg, state, 40.0)
        b = charge_spectrum(cfg, state, 40.0, k_lim=0.5)
        assert a["kx"][1] - a["kx"][0] == pytest.approx(b["kx"][1] - b["kx"][0])

    def test_invalid_window(self):
        cfg = PdwConfig()
        with pytest.raises(ValueError):
            charge_spectrum(cfg, compute_pdw_state(cfg), 0.0)


class TestPdwValidation:
    @pytest.mark.parametrize(
        "overrides",
        [
            {"temperature": -1.0},
            {"temperature": np.nan},
            {"tc": 0.0},
            {"t_pdw": -2.0},
            {"hc2": (20.0, 0.0, 34.0)},
            {"h_pdw": (14.0, 3.75, -1.0)},
            {"h_rot": (0.0, 0.4, 4.5)},
            {"field": (0.0, np.inf, 0.0)},
            {"overlap0": (0.0, 1.5, 1.0)},
            {"overlap0": (-0.1, 0.0, 1.0)},
            {"delta0_mev": -0.1},
            {"pdw_ratio": -0.1},
            {"floor": -1.0},
            {"kappa_q": np.nan},
            {"p_vectors": ((0.1, 0.0), (0.0, 0.1))},
        ],
    )
    def test_invalid_config_raises(self, overrides):
        with pytest.raises(ValueError):
            compute_pdw_state(PdwConfig(**overrides))

    def test_landau_amplitude_invalid_t_star(self):
        with pytest.raises(ValueError):
            landau_amplitude(1.0, 0.0, 0.0)

    def test_helpers_reject_nan_like_the_rust_twin(self):
        with pytest.raises(ValueError):
            landau_amplitude(1.0, np.nan, 0.0)
        with pytest.raises(ValueError):
            reduced_field((1.0, 0.0, 0.0), (np.nan, 1.0, 1.0))
        with pytest.raises(ValueError):
            critical_field_along(np.nan, (1.0, 1.0, 1.0), 0.0, X_AXIS)
        with pytest.raises(ValueError):
            dvector_overlaps((0.0, 0.0, 1.0), (1.0, 1.0, 1.0), (np.nan, 0.0, 1.0))


@pytest.mark.speculative
class TestCrossStackParityAnchors:
    """Pinned values asserted identically in crates/moire-core/src/pdw.rs."""

    def test_q_suppression_field_anchors(self):
        cfg = PdwConfig()
        assert critical_field_along(cfg.t_pdw, cfg.h_pdw, 4.2, X_AXIS) == pytest.approx(
            2.24, abs=1e-12
        )
        assert critical_field_along(cfg.t_pdw, cfg.h_pdw, 4.2, Y_AXIS) == pytest.approx(
            0.6, abs=1e-12
        )
        assert critical_field_along(cfg.t_pdw, cfg.h_pdw, 4.2, Z_AXIS) == pytest.approx(
            6.0, abs=1e-12
        )

    def test_field_induced_p2_amplitude_anchor(self):
        state = _state(temperature=0.3, field=(0.0, 0.0, 6.0))
        assert abs(state.rho_p[1]) == pytest.approx(0.963907488848813, abs=1e-12)

    def test_real_space_origin_anchors(self):
        cfg = PdwConfig()
        pair, charge, spin = pdw_fields_at(cfg, compute_pdw_state(cfg), 0.0, 0.0)
        assert float(pair) == pytest.approx(0.607885168114487, abs=1e-12)
        assert float(charge) == pytest.approx(-12.764863559406859, abs=1e-12)
        assert float(spin) == pytest.approx(7.180927118813718, abs=1e-12)

    def test_moire_locked_twisted_anchors(self):
        # FeTe / Sb2Te3 at a 1 deg twist, T = 4 K, default gaps and xi = 20 A.
        period = moire_locked_period(3.82, 4.264, 1.0)
        assert period == pytest.approx(36.23263757853417, abs=1e-12)
        ratio = moire_locked_pdw_ratio(period, 20.0, 2.58, 3.60)
        assert ratio == pytest.approx(0.047517778919214, abs=1e-12)
        cfg = replace(
            moire_locked_config(period, 1.0, (2.58 + 3.60) / 2.0, ratio), temperature=4.0
        )
        pair, charge, _spin = pdw_fields_at(cfg, compute_pdw_state(cfg), 0.0, 0.0)
        assert float(pair) == pytest.approx(3.331137631734327, abs=1e-12)
        assert float(charge) == pytest.approx(-14.647462277091908, abs=1e-12)
