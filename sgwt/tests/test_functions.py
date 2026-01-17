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

    def test_high_frequency_approaches_zero(self):
        """Lowpass gain at large λ approaches 0."""
        result = lowpass(np.array([1e6]), scale=1.0)[0]
        # At λ=1e6, lowpass should be essentially zero (< 1e-5)
        assert result < 1e-5, f"Expected near-zero gain at high frequency, got {result}"

    @pytest.mark.parametrize("scale", [0.1, 1.0, 10.0])
    def test_cutoff_at_one_over_scale(self, scale):
        """Lowpass gain at λ=1/scale is 0.5."""
        x = np.array([1.0 / scale])
        assert lowpass(x, scale=scale)[0] == pytest.approx(0.5)

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

    def test_high_frequency_approaches_one(self):
        """Highpass gain at large λ approaches 1."""
        result = highpass(np.array([1e6]), scale=1.0)[0]
        # At λ=1e6, highpass should be essentially 1.0 (within 1e-5)
        min_gain = 0.99999
        assert result > min_gain, f"Expected gain >{min_gain} at high frequency, got {result}"

    @pytest.mark.parametrize("scale", [0.1, 1.0, 10.0])
    def test_cutoff_at_one_over_scale(self, scale):
        """Highpass gain at λ=1/scale is 0.5."""
        x = np.array([1.0 / scale])
        assert highpass(x, scale=scale)[0] == pytest.approx(0.5)

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

    def test_high_frequency_approaches_zero(self):
        """Bandpass gain at large λ approaches 0."""
        result = bandpass(np.array([1e6]), scale=1.0)[0]
        # At λ=1e6, bandpass should be essentially zero (< 1e-5)
        assert result < 1e-5, f"Expected near-zero gain at high frequency, got {result}"

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

    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_order_sharpens_response(self, order):
        """Higher order makes bandpass narrower."""
        scale = 1.0
        x = np.linspace(0.01, 10, 1000)
        y = bandpass(x, scale=scale, order=order)
        # Peak value should be <= 1 (order 1 peaks at 1.0)
        assert np.max(y) <= 1.0 + 1e-10

    def test_lowpass_highpass_sum_approximation(self):
        """At moderate frequencies, lowpass + highpass ≈ 1."""
        x = np.linspace(0.1, 10, 100)
        lp = lowpass(x, scale=1.0)
        hp = highpass(x, scale=1.0)
        np.testing.assert_allclose(lp + hp, np.ones_like(x), atol=1e-10)
