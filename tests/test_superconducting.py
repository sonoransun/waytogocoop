"""Tests for superconducting gap modulation and CPDM amplitude."""

from __future__ import annotations

import numpy as np
import pytest

from waytogocoop.computation.superconducting import gap_modulation, cpdm_amplitude
from waytogocoop.config import DELTA_AVG, DELTA_AMPLITUDE


class TestGapModulation:
    def test_output_shape(self):
        pattern = np.random.rand(50, 50)
        gap = gap_modulation(pattern, DELTA_AVG, DELTA_AMPLITUDE)
        assert gap.shape == (50, 50)

    def test_output_range(self):
        """Gap values should stay within [avg - amp, avg + amp]."""
        pattern = np.random.rand(100, 100)
        gap = gap_modulation(pattern, DELTA_AVG, DELTA_AMPLITUDE)
        assert gap.min() >= DELTA_AVG - DELTA_AMPLITUDE - 1e-10
        assert gap.max() <= DELTA_AVG + DELTA_AMPLITUDE + 1e-10

    def test_uniform_input(self):
        """Uniform moire pattern should give uniform gap."""
        pattern = np.full((20, 20), 0.5)
        gap = gap_modulation(pattern, DELTA_AVG, DELTA_AMPLITUDE)
        # All values should be identical
        assert np.allclose(gap, gap[0, 0])

    def test_zero_amplitude(self):
        """Zero modulation amplitude should give constant gap = delta_avg."""
        pattern = np.random.rand(30, 30)
        gap = gap_modulation(pattern, 3.0, 0.0)
        np.testing.assert_allclose(gap, 3.0, atol=1e-12)


class TestCPDMAmplitude:
    def test_large_period_near_one(self):
        """Very large moire period => amplitude close to 1."""
        amp = cpdm_amplitude(1000.0, coherence_length=20.0)
        assert amp > 0.97

    def test_small_period_near_zero(self):
        """Very small moire period => amplitude near 0."""
        amp = cpdm_amplitude(1.0, coherence_length=20.0)
        assert amp < 0.01

    def test_infinite_period_zero(self):
        """Infinite moire period => amplitude 0 (no modulation)."""
        amp = cpdm_amplitude(np.inf)
        assert amp == 0.0

    def test_negative_period_zero(self):
        """Negative period (unphysical) => amplitude 0."""
        amp = cpdm_amplitude(-5.0)
        assert amp == 0.0

    def test_typical_sb2te3_fete(self):
        """Sb2Te3/FeTe moire period ~36.7 A should give reasonable CPDM."""
        amp = cpdm_amplitude(36.7, coherence_length=20.0)
        assert 0.5 < amp < 0.7
