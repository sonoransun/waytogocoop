"""Fourier analysis of moire patterns."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter


def fft_2d(pattern_2d: np.ndarray, dx: float) -> dict:
    """Compute the 2D FFT power spectrum of a real-valued pattern.

    Parameters
    ----------
    pattern_2d : np.ndarray
        2D real-space pattern.
    dx : float
        Grid spacing in real space (Angstrom).

    Returns
    -------
    dict
        ``kx`` (1D, inverse Angstrom), ``ky`` (1D), ``power_spectrum``
        (2D, log10-scaled, fftshift-ed).
    """
    ny, nx = pattern_2d.shape
    ft = np.fft.fft2(pattern_2d)
    ft_shifted = np.fft.fftshift(ft)
    power = np.abs(ft_shifted) ** 2

    # Log scale — add small offset to avoid log(0)
    power_log = np.log10(power + 1.0)

    kx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx)) * 2.0 * np.pi
    ky = np.fft.fftshift(np.fft.fftfreq(ny, d=dx)) * 2.0 * np.pi

    return {
        "kx": kx,
        "ky": ky,
        "power_spectrum": power_log,
    }


def identify_peaks(
    power_spectrum: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray,
    threshold_fraction: float = 0.3,
) -> list[dict]:
    """Identify peaks in a 2D power spectrum above a threshold.

    Parameters
    ----------
    power_spectrum : np.ndarray
        2D (log-scaled) power spectrum.
    kx, ky : np.ndarray
        1D wavevector axes.
    threshold_fraction : float
        Fraction of the maximum value used as the detection threshold.

    Returns
    -------
    list[dict]
        Each entry has keys ``kx``, ``ky``, ``amplitude``.
    """
    threshold = threshold_fraction * power_spectrum.max()

    # Local maximum detection with a small neighbourhood
    neighbourhood_size = max(5, min(power_spectrum.shape) // 20)
    local_max = maximum_filter(power_spectrum, size=neighbourhood_size)
    is_peak = (power_spectrum == local_max) & (power_spectrum > threshold)

    # Exclude the DC component (centre of the spectrum)
    cy, cx = power_spectrum.shape[0] // 2, power_spectrum.shape[1] // 2
    margin = max(2, neighbourhood_size // 2)
    is_peak[cy - margin : cy + margin + 1, cx - margin : cx + margin + 1] = False

    peak_indices = np.argwhere(is_peak)  # (row, col)
    peaks: list[dict] = []
    for row, col in peak_indices:
        peaks.append(
            {
                "kx": float(kx[col]),
                "ky": float(ky[row]),
                "amplitude": float(power_spectrum[row, col]),
            }
        )
    # Sort by descending amplitude
    peaks.sort(key=lambda p: p["amplitude"], reverse=True)
    return peaks
