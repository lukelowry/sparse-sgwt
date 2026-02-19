# -*- coding: utf-8 -*-
"""
Tests for analytical filter functions (functions.analytical module).
"""
import numpy as np
import pytest

from sgwt.functions import lowpass, highpass, bandpass


class TestLowpass:
    """Tests for lowpass filter function."""

    def test_dc_gain_is_one(self):
        """Lowpass gain at λ=0 is 1."""
        assert lowpass(np.array([0.0]), scale=1.0)[0] == pytest.approx(1.0)

    def test_monotonically_decreasing(self):
        """Lowpass is monotonically decreasing for λ > 0."""
        x = np.linspace(0, 10, 100)
        y = lowpass(x, scale=1.0)
        assert np.all(np.diff(y) <= 0)


class TestHighpass:
    """Tests for highpass filter function."""

    def test_dc_gain_is_zero(self):
        """Highpass gain at λ=0 is 0."""
        assert highpass(np.array([0.0]), scale=1.0)[0] == pytest.approx(0.0)

    def test_monotonically_increasing(self):
        """Highpass is monotonically increasing for λ > 0."""
        x = np.linspace(0, 10, 100)
        y = highpass(x, scale=1.0)
        assert np.all(np.diff(y) >= 0)


class TestBandpass:
    """Tests for bandpass filter function."""

    def test_dc_gain_is_zero(self):
        """Bandpass gain at λ=0 is 0."""
        assert bandpass(np.array([0.0]), scale=1.0)[0] == pytest.approx(0.0)

    def test_peak_at_center_frequency(self):
        """Bandpass has maximum near center frequency λ=1/scale."""
        scale = 1.0
        x = np.linspace(0.01, 10, 1000)
        y = bandpass(x, scale=scale)
        peak_idx = np.argmax(y)
        peak_x = x[peak_idx]
        # Peak should be near 1/scale, within 20% tolerance
        tolerance_fraction = 0.2  # Allow 20% deviation from expected peak location
        expected_peak = 1.0 / scale
        deviation = abs(peak_x - expected_peak)
        assert deviation < tolerance_fraction, \
            f"Peak at λ={peak_x:.3f}, expected near λ={expected_peak:.3f} (tolerance={tolerance_fraction})"
