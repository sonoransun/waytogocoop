"""Tests for 2D FFT power spectrum and peak identification."""

from __future__ import annotations

import numpy as np
import pytest

from waytogocoop.computation.fourier import fft_2d, identify_peaks


class TestFFT2D:
    def test_output_keys(self):
        """fft_2d should return dict with kx, ky, power_spectrum."""
        pattern = np.random.default_rng(42).random((64, 64))
        result = fft_2d(pattern, dx=1.0)
        assert set(result.keys()) == {"kx", "ky", "power_spectrum"}

    def test_output_shapes(self):
        """For NxN input, kx and ky have length N and power_spectrum is (N, N)."""
        N = 128
        pattern = np.random.default_rng(0).random((N, N))
        result = fft_2d(pattern, dx=1.0)
        assert result["kx"].shape == (N,)
        assert result["ky"].shape == (N,)
        assert result["power_spectrum"].shape == (N, N)

    def test_uniform_input_dc_peak(self):
        """Uniform input concentrates all power at DC (center of shifted spectrum)."""
        N = 64
        pattern = np.ones((N, N)) * 5.0
        result = fft_2d(pattern, dx=1.0)
        ps = result["power_spectrum"]

        center_y, center_x = N // 2, N // 2
        # The DC component should be the global maximum
        assert ps[center_y, center_x] == pytest.approx(ps.max())

    def test_power_spectrum_non_negative(self):
        """Log-scaled power log10(|F|^2 + 1) should be >= 0 everywhere."""
        pattern = np.random.default_rng(7).random((64, 64))
        result = fft_2d(pattern, dx=1.0)
        assert np.all(result["power_spectrum"] >= 0.0)

    def test_sinusoidal_input_produces_peaks(self):
        """cos(2*pi*f*x) on a 2D grid should produce peaks near expected kx values."""
        N = 128
        dx = 1.0
        freq = 5  # 5 cycles across the grid
        x = np.arange(N) * dx
        y = np.arange(N) * dx
        X, Y = np.meshgrid(x, y)

        # Spatial frequency in cycles per unit length
        f_spatial = freq / (N * dx)
        # Corresponding angular wavenumber k = 2*pi*f_spatial
        k_expected = 2.0 * np.pi * f_spatial

        pattern = np.cos(2.0 * np.pi * f_spatial * X)
        result = fft_2d(pattern, dx=dx)

        ps = result["power_spectrum"]
        kx = result["kx"]

        # Find the row at ky=0 (center row) and look for peaks along kx
        center_row = N // 2
        kx_slice = ps[center_row, :]

        # The two highest non-DC peaks should be near +/- k_expected
        # Exclude the DC bin, find the two brightest kx positions
        dc_col = N // 2
        kx_slice_no_dc = kx_slice.copy()
        kx_slice_no_dc[dc_col] = 0.0

        peak_indices = np.argsort(kx_slice_no_dc)[-2:]
        peak_kx_values = np.sort(np.abs(kx[peak_indices]))

        # Both peaks should be at |kx| ~ k_expected
        np.testing.assert_allclose(peak_kx_values, k_expected, atol=0.15)

    def test_kx_ky_scaling_with_dx(self):
        """kx max should scale as approximately pi/dx (Nyquist frequency)."""
        N = 64
        for dx in [0.5, 1.0, 2.0]:
            result = fft_2d(np.zeros((N, N)), dx=dx)
            kx = result["kx"]
            expected_max = np.pi / dx
            # The maximum kx should be close to pi/dx
            # For even N the positive Nyquist bin is at pi/dx, but fftfreq
            # may place the folded bin at -pi/dx, so check the absolute max.
            assert np.max(np.abs(kx)) == pytest.approx(expected_max, rel=0.05)


class TestIdentifyPeaks:
    def test_dc_component_excluded(self):
        """For uniform input DC is the only peak; after exclusion the list is empty."""
        N = 64
        pattern = np.ones((N, N)) * 3.0
        result = fft_2d(pattern, dx=1.0)
        peaks = identify_peaks(
            result["power_spectrum"], result["kx"], result["ky"]
        )
        assert peaks == []

    def test_sinusoidal_peak_positions(self):
        """Identified peaks for a known sinusoid should match expected (kx, ky)."""
        N = 128
        dx = 1.0
        freq = 8  # cycles across the grid
        f_spatial = freq / (N * dx)
        k_expected = 2.0 * np.pi * f_spatial

        x = np.arange(N) * dx
        y = np.arange(N) * dx
        X, Y = np.meshgrid(x, y)
        pattern = np.cos(2.0 * np.pi * f_spatial * X)
        result = fft_2d(pattern, dx=dx)

        peaks = identify_peaks(
            result["power_spectrum"],
            result["kx"],
            result["ky"],
            threshold_fraction=0.1,
        )

        assert len(peaks) >= 2

        # The strongest peaks should lie near ky=0 and kx ~ +/- k_expected
        peak_kx_values = sorted([abs(p["kx"]) for p in peaks[:2]])
        peak_ky_values = [abs(p["ky"]) for p in peaks[:2]]

        np.testing.assert_allclose(peak_kx_values, k_expected, atol=0.2)
        np.testing.assert_allclose(peak_ky_values, 0.0, atol=0.2)

    def test_peaks_sorted_by_amplitude(self):
        """Returned peaks should be sorted in descending amplitude order."""
        N = 128
        dx = 1.0
        x = np.arange(N) * dx
        y = np.arange(N) * dx
        X, Y = np.meshgrid(x, y)

        # Two cosines at different frequencies to get multiple peaks
        f1 = 4 / (N * dx)
        f2 = 12 / (N * dx)
        pattern = np.cos(2.0 * np.pi * f1 * X) + 0.5 * np.cos(2.0 * np.pi * f2 * Y)
        result = fft_2d(pattern, dx=dx)

        peaks = identify_peaks(
            result["power_spectrum"],
            result["kx"],
            result["ky"],
            threshold_fraction=0.05,
        )

        if len(peaks) >= 2:
            amplitudes = [p["amplitude"] for p in peaks]
            assert amplitudes == sorted(amplitudes, reverse=True)

    def test_high_threshold_reduces_peaks(self):
        """A higher threshold_fraction should return fewer (or equal) peaks."""
        N = 128
        dx = 1.0
        x = np.arange(N) * dx
        y = np.arange(N) * dx
        X, Y = np.meshgrid(x, y)

        # Multi-frequency pattern to produce several peaks
        f1 = 5 / (N * dx)
        f2 = 15 / (N * dx)
        f3 = 10 / (N * dx)
        pattern = (
            np.cos(2.0 * np.pi * f1 * X)
            + 0.6 * np.cos(2.0 * np.pi * f2 * Y)
            + 0.3 * np.cos(2.0 * np.pi * f3 * (X + Y))
        )
        result = fft_2d(pattern, dx=dx)

        peaks_low = identify_peaks(
            result["power_spectrum"],
            result["kx"],
            result["ky"],
            threshold_fraction=0.1,
        )
        peaks_high = identify_peaks(
            result["power_spectrum"],
            result["kx"],
            result["ky"],
            threshold_fraction=0.95,
        )

        assert len(peaks_high) <= len(peaks_low)
